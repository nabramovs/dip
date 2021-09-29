import os
import sys
import pymongo
import csv
import numpy as np
from progressbar import ProgressBar, ETA, Percentage, Bar
from collections import OrderedDict
from common import MongoDB, write_table_to_csv
from gnomad_utils import worst_csq_from_csq
from features import add_feature

PROGRESS_BAR_WIDGETS = [Percentage(), ' ', Bar(), ' ', ETA()]

class SplicedExon():
	def __init__(self):
		self.hgnc_id = ''
		self.transcript_id = ''
		self.start = 0
		self.xstart = 0
		self.stop = 0
		self.xstop = 0
		self.strand = ''
		self.chrom = ''
		self.splicing_variants = []
		self.single_exon = False
		self.exon_length = 0
		self.exon_relative_length = 0

	def get_dictionary(self):
		dictionary = OrderedDict()
		dictionary['hgnc_gene_id'] = self.hgnc_id
		dictionary['transcript_id'] = self.transcript_id
		dictionary['start'] = self.start
		dictionary['xstart'] = self.xstart
		dictionary['stop'] = self.stop
		dictionary['xstop'] = self.xstop
		dictionary['strand'] = self.strand
		dictionary['chrom'] = self.chrom
		dictionary['splicing_variants'] = self.splicing_variants
		dictionary['single_exon'] = self.single_exon
		dictionary['exon_length'] = self.exon_length
		dictionary['exon_relative_length'] = self.exon_relative_length
		return dictionary


# Get only High Confidence (LOFTEE) splice acceptor variants
def get_splice_acceptor_variants(db, xstart, xstop, transcript_id, exomes=True):
	splice_acceptor_variants = set([])
	if exomes:
		variant_collection = 'exome_variants'
	else:
		variant_collection = 'genome_variants'

	variants = db.exac[variant_collection].find({ "$and": [ { "xpos": { "$gte": xstart } }, { "xpos": { "$lte": xstop } } ] })
	for variant in variants:
		if 'vep_annotations' not in variant:
			continue
		for vep in variant['vep_annotations']:
			csq = worst_csq_from_csq(vep['Consequence'])
			if csq == 'splice_acceptor_variant' and vep['Feature'] == transcript_id and vep['LoF'] == 'HC':
				splice_acceptor_variants.add(variant['variant_id'])
	return splice_acceptor_variants


def get_relative_exon_lengths(db, transcript_ids):
	transcripts_all_exons_length = {}

	for transcript_id in transcript_ids:
		total_length = 0
		exons = db.exac.exons.find({ "transcript_id": transcript_id, "feature_type": "CDS" }).sort([("xstart", pymongo.ASCENDING)])
		for exon in exons:
			# Add one to include exon's first base (e.g. stop-start = 5-1 amino acids = 4, but real length = 5)
			exon_length = exon['stop'] - exon['start'] + 1
			total_length += exon_length
		if total_length > 0:
			transcripts_all_exons_length[transcript_id] = total_length

	transcripts_exons_relative_length = {}
	for transcript_id, transcript_length in transcripts_all_exons_length.items():
		exons_relative_length = OrderedDict()
		exons = db.exac.exons.find({ "transcript_id": transcript_id, "feature_type": "CDS" }).sort([("xstart", pymongo.ASCENDING)])
		for exon in exons:
			exon_length = exon['stop'] - exon['start'] + 1
			exons_relative_length[exon['start']] = float(exon_length) / transcript_length

		transcripts_exons_relative_length[transcript_id] = exons_relative_length

	return transcripts_exons_relative_length


def create_spliced_exons(db):
	db.main.spliced_exons.drop()

	hgnc_id_to_canonical_transcript = db.main.mapping_dicts.find_one({'_id': 'hgnc_id_to_canonical_transcript'})
	hgnc_id_to_canonical_transcript = hgnc_id_to_canonical_transcript['mapping_dict']
	transcript_ids = hgnc_id_to_canonical_transcript.values()
	transcripts_exons_relative_length = get_relative_exon_lengths(db, transcript_ids)

	total_lines = len(transcripts_exons_relative_length)
	line_num = 0
	bar = ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()
	for hgnc_id, transcript_id in hgnc_id_to_canonical_transcript.items():
		# Skip transcripts without coding exons in gnomAD
		if transcript_id not in transcripts_exons_relative_length:
			continue
		exons_relative_length = transcripts_exons_relative_length[transcript_id]

		exons = db.exac.exons.find({ "transcript_id": transcript_id, "feature_type": "CDS" })
		exons_num = db.exac.exons.count_documents({ "transcript_id": transcript_id, "feature_type": "CDS" })
		for exon in exons:
			if exon['strand'] == '+':
				splice_xstart = exon['xstart'] - 5
				splice_xstop = exon['xstart'] + 5
			else:
				splice_xstart = exon['xstop'] - 5
				splice_xstop = exon['xstop'] + 5

			splice_acceptor_variants = set([])
			splice_acceptor_variants |= get_splice_acceptor_variants(db, splice_xstart, splice_xstop, transcript_id, exomes=True)
			splice_acceptor_variants |= get_splice_acceptor_variants(db, splice_xstart, splice_xstop, transcript_id, exomes=False)

			if len(splice_acceptor_variants) > 0:
				spliced_exon = SplicedExon()

				spliced_exon.hgnc_id = hgnc_id
				spliced_exon.transcript_id = transcript_id
				spliced_exon.start = exon['start']
				spliced_exon.xstart = exon['xstart']
				spliced_exon.stop = exon['stop']
				spliced_exon.xstop = exon['xstop']
				spliced_exon.strand = exon['strand']
				spliced_exon.chrom = exon['chrom']
				spliced_exon.splicing_variants = list(splice_acceptor_variants)

				if exons_num == 1:
					spliced_exon.single_exon = True
				else:
					spliced_exon.single_exon = False

				spliced_exon.exon_length = exon['stop'] - exon['start'] + 1
				spliced_exon.exon_relative_length = exons_relative_length[exon['start']]

				db.main.spliced_exons.insert_one(spliced_exon.get_dictionary())

		line_num += 1
		bar.update((line_num + 0.0) / total_lines)
	bar.finish()

	db.main.spliced_exons.create_index([('hgnc_id', pymongo.ASCENDING)], name='hgnc_id_1')
	db.main.spliced_exons.create_index([('transcript_id', pymongo.ASCENDING)], name='transcript_id_1')


def create_splice_acceptor_gene_feature(db):
	# Calculate splice acceptor feature for genes with spliced exons
	spliced_exons = db.main.spliced_exons.find({"single_exon": False })
	spliced_genes = {}
	for spliced_exon in spliced_exons:
		hgnc_id = spliced_exon['hgnc_gene_id']
		if hgnc_id not in spliced_genes:
			spliced_genes[hgnc_id] = spliced_exon['exon_relative_length']
		else:
			spliced_genes[hgnc_id] += spliced_exon['exon_relative_length']

	# Get HGNC gene ids, which were mapped to gnomAD canonical transcripts
	hgnc_id_to_canonical_transcript = db.main.mapping_dicts.find_one({'_id': 'hgnc_id_to_canonical_transcript'})
	hgnc_id_to_canonical_transcript = hgnc_id_to_canonical_transcript['mapping_dict']
	# Get transcript IDs which had coding exons in gnomAD
	transcript_ids = list(hgnc_id_to_canonical_transcript.values())
	transcripts_exons_relative_length = get_relative_exon_lengths(db, transcript_ids)

	# Calculate splice acceptor feature for all genes which were analysed (i.e. had coding exons)
	# Note that 0.0 feature value means that there were splicing variants (i.e. gene might be important!),
	# that's why gene without coding exons in gnomAD were skipped and not assigned 0.0 values.
	splice_acceptor_feature = {}
	for hgnc_id, transcript_id in hgnc_id_to_canonical_transcript.items():
		# Skip transcripts without coding exons in gnomAD
		if transcript_id not in transcripts_exons_relative_length:
			continue
		
		if hgnc_id in spliced_genes:
			splice_acceptor_feature[hgnc_id] = spliced_genes[hgnc_id]
		else:
			splice_acceptor_feature[hgnc_id] = 0.0

	add_feature(db, 'splice_acceptor', 'median', np.median(list(splice_acceptor_feature.values())), 
		        splice_acceptor_feature)


def main():
	db = MongoDB()
	create_spliced_exons(db)
	create_splice_acceptor_gene_feature(db)


if __name__ == "__main__":
	sys.exit(main())