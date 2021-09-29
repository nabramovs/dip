import os
import sys
import numpy as np
from collections import OrderedDict
from common import MongoDB
from gene_groups import GeneGroups


class Features():
	def __init__(self, db):
		self.db = db
		self.data = {}
		self.defaults = {}
		features = self.db.main.features.find({})

		for feature in features:
			name = feature['_id']
			self.data[name] = feature['data']
			self.defaults[name] = feature['default_value']


	def get_values(self, hgnc_ids, feature_name_groups):
		# Split joint features (+)
		feature_names = []

		for feature_name_group in feature_name_groups:
			feature_names += feature_name_group.split('+')

		table = []
		for hgnc_id in hgnc_ids:
			row = []
			for feature_name in feature_names:
				if hgnc_id in self.data[feature_name]:
					row.append(self.data[feature_name][hgnc_id])
				else:
					row.append(self.defaults[feature_name])
			table.append(row)
		return table


	def get_gene_id_to_value_dict(self, hgnc_ids, feature_name):
		gene_id_to_value = OrderedDict()
		for hgnc_id in hgnc_ids:
			if hgnc_id in self.data[feature_name]:
				gene_id_to_value[hgnc_id] = self.data[feature_name][hgnc_id]
			else:
				gene_id_to_value[hgnc_id] = self.defaults[feature_name]
		return gene_id_to_value


def add_feature(db, feature_id, default_type, default_value, data, stats=None):
	feature = OrderedDict()
	feature['_id'] = feature_id
	if stats:
		feature['stats'] = stats
	feature['default_type'] = default_type
	feature['default_value'] = default_value
	feature['data'] = data

	db.main.features.delete_one({'_id': feature['_id']})
	db.main.features.insert_one(feature)


#######################
### CREATE FEATURES ###
#######################

def create_gevir_loeuf_virlof_features(db):
	gevir = {}
	loeuf = {}
	virlof = {}

	gevir_genes = db.main.gevir_genes.find({})

	for gevir_gene in gevir_genes:
		hgnc_id = gevir_gene['hgnc_gene_id']
		gevir[hgnc_id] = gevir_gene['gevir']
		loeuf[hgnc_id] = gevir_gene['loeuf']
		virlof[hgnc_id] = gevir_gene['virlof']

	add_feature(db, 'gevir', 'median', np.median(list(gevir.values())), gevir)
	add_feature(db, 'loeuf', 'median', np.median(list(loeuf.values())), loeuf)
	add_feature(db, 'virlof', 'median', np.median(list(virlof.values())), virlof)


def create_domino_features(db):
	domino_splice_donor = {}
	domino_5_prime_utr_conservation = {}
	domino_mrna_half_life_gt_10h = {}

	domino_genes = db.main.domino_genes.find({})
	for domino_gene in domino_genes:
		hgnc_id = domino_gene['hgnc_gene_id']
		domino_splice_donor[hgnc_id] = domino_gene['n_donor_divided_by_n_synonymous']
		domino_5_prime_utr_conservation[hgnc_id] = domino_gene['phylo_pat_5_prime_utr']
		domino_mrna_half_life_gt_10h[hgnc_id] = domino_gene['mrna_half_life_gt_10h']

	add_feature(db, 'domino_splice_donor', 'median', np.median(list(domino_splice_donor.values())), domino_splice_donor)
	add_feature(db, 'domino_5_prime_utr_conservation', 'median', 
				np.median(list(domino_5_prime_utr_conservation.values())), domino_5_prime_utr_conservation)
	add_feature(db, 'domino_mrna_half_life_gt_10h', 'false', 0, domino_mrna_half_life_gt_10h)


def create_uneecon_feature(db):
	uneecon_genes = db.main.uneecon_genes.find({})
	data = {}
	for uneecon_gene in uneecon_genes:
		data[uneecon_gene['hgnc_gene_id']] = uneecon_gene['score']

	add_feature(db, 'uneecon', 'median', np.median(list(data.values())), data)


def create_gnomad_lof_sv_features(db):
	gnomad_lof_svs = db.main.gnomad_control_only_svs.find({ "svtype": "DEL", 
														    "FILTER": "PASS", 
														    "lof_hgnc_gene_ids.0": { "$exists": True } })
	gene_id_to_sum_af = {}

	for gnomad_lof_sv in gnomad_lof_svs:
		gene_ids = gnomad_lof_sv['lof_hgnc_gene_ids']
		af = float(gnomad_lof_sv["AF"])
		for gene_id in gene_ids:
			if gene_id not in gene_id_to_sum_af:
				gene_id_to_sum_af[gene_id] = af
			else:
				gene_id_to_sum_af[gene_id] += af

	add_feature(db, 'gnomad_sv_lof_sum_af', 'no_data', 0, gene_id_to_sum_af)


def main():
	db = MongoDB()

	# Uncomment to create features
	#create_gevir_loeuf_virlof_features(db)
	#create_domino_features(db)
	#create_uneecon_feature(db)
	#create_gnomad_lof_sv_features(db)

if __name__ == "__main__":
	sys.exit(main())