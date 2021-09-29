import os
import sys
import pymongo
import csv
import json
from pathlib import Path
from progressbar import ProgressBar, ETA, Percentage, Bar
from collections import OrderedDict
from common import MongoDB, write_table_to_csv, file_len
from csv_reader import CsvReader

PROGRESS_BAR_WIDGETS = [Percentage(), ' ', Bar(), ' ', ETA()]

SOURCE_DATA_FOLDER = Path('source_data/')

# https://www.genenames.org/download/statistics-and-files/ (03.03.20)
HGNC_GENES_JSON = SOURCE_DATA_FOLDER / 'hgnc_protein-coding_gene.json'

# GDIT: supplementary data from here: https://www.nature.com/articles/s41525-019-0081-z
GDIT_CSV = SOURCE_DATA_FOLDER / 'gdit_genes.csv'

# ENSEMBL Build 37 gene-transcript-protein ids to HGNC name and id mapping file
# obtained from ENSEMBL BioMart (06.12.19)
ENS_PROTEINS_37_TSV = SOURCE_DATA_FOLDER / 'ens_proteins_37.tsv'

# STRING: data obtained from here: https://string-db.org/
STRING_V10_TXT = SOURCE_DATA_FOLDER / '9606.protein.links.detailed.v10.txt' # v10 used by DOMINO

# GeVIR: supplementary data from here: https://www.nature.com/articles/s41588-019-0560-2
GEVIR_LOEUF_VIRLOF_SCORES_CSV = SOURCE_DATA_FOLDER / 'gevir_loeuf_virlof_scores.csv'

# UNEECON Gene scores (https://psu.app.box.com/s/wur3td0dawju9qtvu7w8orkxu5ur0oo6/file/517942997406) version 03.09.19
UNEECON_G_TSV = SOURCE_DATA_FOLDER / 'UNEECON-G_gene_score_v1.0_hg19.tsv'

# DOMINO v1 final, train and validation (from package) gene scores were obtained from the website 
# (https://wwwfbm.unil.ch/domino/download.html) (version: 19.02.19)
DOMINO_FINAL_GENE_SCORES_TSV = SOURCE_DATA_FOLDER / 'domino_final.tsv'
DOMINO_TRAIN_GENES_TSV = SOURCE_DATA_FOLDER / 'domino_train.tsv'
DOMINO_VALIDATION_GENES_TSV = SOURCE_DATA_FOLDER / 'domino_validation.tsv'

# GnomAD Structural Variants v2.1.1 https://gnomad.broadinstitute.org/downloads (accessed 06.01.20)
GNOMAD_CONTROL_ONLY_SVS = SOURCE_DATA_FOLDER / 'gnomad_v2.1_sv.controls_only.sites.bed'

# GPP Supplementary Material 8 (Table 4) scores from:
# Gene pathogenicity prediction of Mendelian diseases via the Random Forest algorithm
# https://doi.org/10.1007/s00439-019-02021-9
GPP_GENE_SCORES_CSV = SOURCE_DATA_FOLDER / 'gpp_gene_scores.csv'
GDP_GENE_SCORES_CSV = SOURCE_DATA_FOLDER / 'gdp_gene_scores.csv'
GRP_GENE_SCORES_CSV = SOURCE_DATA_FOLDER / 'grp_gene_scores.csv'

GPP_TRAIN_DISEASE_LIST = SOURCE_DATA_FOLDER / 'gpp_train_disease.tsv'
GPP_TRAIN_TOLERANT_LIST = SOURCE_DATA_FOLDER / 'gpp_train_tolerant.tsv'

# Gene4Denovo (25.02.20)
# http://www.genemed.tech/gene4denovo/uploads/Candidate_gene.txt
# https://academic.oup.com/nar/article/48/D1/D913/5603227
GENE4DENOVO_TSV = SOURCE_DATA_FOLDER / 'gene4denovo.tsv'

# From MacArthur gene_lists (accessed 06.01.20)
# https://github.com/macarthur-lab/gene_lists/

# Mainland 2015 "Human olfactory receptor responses to odorants"
# https://www.nature.com/articles/sdata20152
OLFACTORY_GENE_LIST = SOURCE_DATA_FOLDER / 'mac_arthur_olfactory_receptors.tsv'

# Hart et al 2017 "Evaluation and Design of Genome-Wide CRISPR/SpCas9 Knockout Screens"
# https://www.g3journal.org/content/7/8/2719
CELL_ESSENTIAL_GENE_LIST = SOURCE_DATA_FOLDER / 'mac_arthur_essential.tsv'
CELL_NON_ESSENTIAL_GENE_LIST = SOURCE_DATA_FOLDER / 'mac_arthur_non_essential.tsv'

# From "Regional missense constraint improves variant deleteriousness prediction"
# Supplementary Table S5a
SEVERE_HI_GENE_LIST = SOURCE_DATA_FOLDER / 'samocha_severe_hi.tsv'


#################
### HUGO/HGNC ###
#################

def import_hgnc_genes(db):
	# HGNC_GENES_JSON
	genes = []
	with open(HGNC_GENES_JSON, 'r') as f:
		# Note: HGNC json contains only 1 line...
		for line in f:
			data = json.loads(line)['response']['docs']
			# process genes data
			for gene in data:
				processed_gene = OrderedDict()
				for k, v in gene.items():
					if '.' in k:
						k = k.replace('.', '_dot_')
					processed_gene[k] = v

				genes.append(processed_gene)

	db.main.hgnc_genes.drop()
	db.main.hgnc_genes.insert_many(genes)
	db.main.hgnc_genes.create_index([('hgnc_id', pymongo.ASCENDING)], name='hgnc_id_1')
	db.main.hgnc_genes.create_index([('symbol', pymongo.ASCENDING)], name='symbol_1')
	db.main.hgnc_genes.create_index([('prev_symbol', pymongo.ASCENDING)], name='prev_symbol_1')


def get_hgnc_gene_dicts(db):
	gene_id_to_name = {}
	gene_name_to_id = {}

	seen_alias_gene_names = set()
	alias_gene_name_to_id = {}

	seen_prev_gene_names = set()
	prev_gene_name_to_id = {}

	hgnc_genes = db.main.hgnc_genes.find({})
	for hgnc_gene in hgnc_genes:
		gene_name = hgnc_gene['symbol']
		gene_id = hgnc_gene['hgnc_id']
		gene_name_to_id[gene_name] = gene_id
		gene_id_to_name[gene_id] = gene_name

		if 'alias_symbol' in hgnc_gene:
			for alias_gene_name in hgnc_gene['alias_symbol']:
				if alias_gene_name in seen_alias_gene_names:
					# delete duplicate and don't insert new link
					del alias_gene_name_to_id[alias_gene_name]
				else:
					alias_gene_name_to_id[alias_gene_name] = gene_id

		if 'prev_symbol' in hgnc_gene:
			for prev_gene_name in hgnc_gene['prev_symbol']:
				if prev_gene_name in seen_prev_gene_names:
					# delete duplicate and don't insert new link
					del prev_gene_name_to_id[prev_gene_name]
				else:
					prev_gene_name_to_id[prev_gene_name] = gene_id

	official_gene_names = set(gene_name_to_id.keys())
	alias_gene_names = set(alias_gene_name_to_id.keys())
	prev_gene_names = set(prev_gene_name_to_id.keys())

	alt_gene_names = alias_gene_names | prev_gene_names
	prev_and_alias_gene_names = alias_gene_names & prev_gene_names

	# Exclude gene names which are present in both alias and previous lists, 
	# but link to different gene_ids 
	for gene_name in prev_and_alias_gene_names:
		if alias_gene_name_to_id[gene_name] != prev_gene_name_to_id[gene_name]:
			alt_gene_names.remove(gene_name)

	# Exclude gene names which are present in current official list
	alt_gene_names -= official_gene_names

	alt_gene_name_to_id = {}
	for alt_gene_name in alt_gene_names:
		if alt_gene_name in alias_gene_names:
			alt_gene_name_to_id[alt_gene_name] = alias_gene_name_to_id[alt_gene_name]
		else:
			alt_gene_name_to_id[alt_gene_name] = prev_gene_name_to_id[alt_gene_name]

	return gene_name_to_id, alt_gene_name_to_id, gene_id_to_name


def remove_dots_from_gene_name_list(gene_names):
	no_dots_gene_names = []
	for gene_name in gene_names:
		if '.' in gene_name:
			gene_name = gene_name.replace('.', '_dot_')
		no_dots_gene_names.append(gene_name)
	return no_dots_gene_names


def remove_dots_from_gene_name_dict_keys(gene_names):
	no_dots_gene_names = OrderedDict()
	for gene_name, value in gene_names.items():
		if '.' in gene_name:
			gene_name = gene_name.replace('.', '_dot_')
		no_dots_gene_names[gene_name] = value
	return no_dots_gene_names


def get_mapped_gene_name_to_hgnc_dicts(db, gene_list_name, gene_names, check_for_duplicates=True, store_report=True):
	# Mapped dicts for output
	mapped_gene_names = {}
	mapped_gene_ids = {}

	gene_name_to_id, alt_gene_name_to_id, gene_id_to_name = get_hgnc_gene_dicts(db)

	if check_for_duplicates:
		seen_gene_names = set()
		duplicate_gene_names = set()

		for gene_name in gene_names:
			if gene_name in seen_gene_names:
				duplicate_gene_names.add(gene_name)
			else:
				seen_gene_names.add(gene_name)

		# Remove duplicates from the list
		gene_names_set = seen_gene_names - duplicate_gene_names
	else:
		gene_names_set = set(gene_names)

	# Map gene name to id using official gene names
	used_gene_ids = set()
	not_found_gene_names = set()
	found_by_off_gene_names = []

	for gene_name in gene_names_set:
		if gene_name in gene_name_to_id:
			gene_id = gene_name_to_id[gene_name]
			used_gene_ids.add(gene_id)
			mapped_gene_ids[gene_name] = gene_id
			mapped_gene_names[gene_name] = gene_id_to_name[gene_id]
			found_by_off_gene_names.append(gene_name)
		else:
			not_found_gene_names.add(gene_name)

	# Find which gene names can be mapped to ids using alternative gene names,
	# BUT EXCLUDE genes which can be mapped to gene id already used by official gene name
	# OR multiple gene names can be mapped to the same gene id
	seen_alt_gene_ids = set()
	duplicate_alt_gene_ids = set()
	for gene_name in not_found_gene_names:
		if gene_name in alt_gene_name_to_id:
			gene_id = alt_gene_name_to_id[gene_name]
			if gene_id not in used_gene_ids:
				if gene_id in seen_alt_gene_ids:
					duplicate_alt_gene_ids.add(gene_id)
				else:
					seen_alt_gene_ids.add(gene_id)

	passed_gene_ids = seen_alt_gene_ids - duplicate_alt_gene_ids
	# Map gene name to id using alternative gene names
	found_by_alt_gene_names = []
	for gene_name in not_found_gene_names:
		if gene_name in alt_gene_name_to_id:
			gene_id = alt_gene_name_to_id[gene_name]
			if gene_id in passed_gene_ids:
				mapped_gene_ids[gene_name] = gene_id
				mapped_gene_names[gene_name] = gene_id_to_name[gene_id]
				found_by_alt_gene_names.append(gene_name)

	# Find genes that were not mapped by official or alternative gene name
	not_found_gene_names -= set(found_by_alt_gene_names)

	# CHECKS, if the code works correctly, these conditions cannot be satisfied
	if len(set(mapped_gene_ids.keys())) != len(set(mapped_gene_ids.values())):
		print('BUG: multiple genes were mapped to the same gene id')
		print('DEBUGGING:')
		print('Gene names:', len(set(mapped_gene_ids.keys())))
		print('Gene ids:', len(set(mapped_gene_ids.values())))

		seen_gene_ids = set()
		duplicate_gene_ids = set()
		for gene_name, gene_id in mapped_gene_ids.items():
			if gene_id in seen_gene_ids:
				duplicate_gene_ids.add(gene_id)
			else:
				seen_gene_ids.add(gene_id)

		for gene_name, gene_id in mapped_gene_ids.items():
			if gene_id in duplicate_gene_ids:
				print(gene_id, gene_name)

	if len(gene_names_set) != len(mapped_gene_names) + len(not_found_gene_names):
		print('BUG: some genes were somehow lost in the mapping process')

	# Report
	if store_report:
		mapping_report = OrderedDict()
		mapping_report['_id'] = gene_list_name
		if check_for_duplicates:
			mapping_report['duplicate_gene_names'] = remove_dots_from_gene_name_list(list(duplicate_gene_names)) 
			mapping_report['duplicate_gene_names_num'] = len(duplicate_gene_names)
		mapping_report['original_unique_genes'] = list(gene_names_set)
		mapping_report['original_unique_genes_num'] = len(gene_names_set)
		mapping_report['found_by_off_gene_names'] = remove_dots_from_gene_name_list(found_by_off_gene_names)
		mapping_report['found_by_off_gene_names_num'] = len(found_by_off_gene_names)
		mapping_report['found_by_alt_gene_names'] = remove_dots_from_gene_name_list(found_by_alt_gene_names)
		mapping_report['found_by_alt_gene_names_num'] = len(found_by_alt_gene_names)
		mapping_report['not_found_gene_names'] = remove_dots_from_gene_name_list(list(not_found_gene_names))
		mapping_report['not_found_gene_names_num'] = len(not_found_gene_names)
		mapping_report['mapped_gene_names'] = remove_dots_from_gene_name_dict_keys(mapped_gene_names)
		mapping_report['mapped_gene_ids'] = remove_dots_from_gene_name_dict_keys(mapped_gene_ids)

		db.main.gene_mapping_reports.delete_one({'_id': gene_list_name})
		db.main.gene_mapping_reports.insert_one(mapping_report)

	return mapped_gene_names, mapped_gene_ids


def import_gene_collection_with_csv_reader_and_hgnc_data(db, file_path, gene_name_column_name, collection_name, 
	                                                     delimiter=',', columns_dict={}, store_report=True,
	                                                     check_for_duplicates=True, csv_reader=None):
	if not csv_reader:
		csv_reader = CsvReader(file_path, delimiter=delimiter)

	gene_names = []

	for document in csv_reader.data:
		gene_name = document[gene_name_column_name]
		gene_names.append(gene_name)

	mapped_gene_names, mapped_gene_ids = get_mapped_gene_name_to_hgnc_dicts(db, collection_name, gene_names, store_report=store_report,
		                                                                    check_for_duplicates=check_for_duplicates)
	updated_data = []
	for document in csv_reader.data:
		gene_name = document[gene_name_column_name]
		if gene_name in mapped_gene_names:
			updated_document = OrderedDict()
			updated_document['hgnc_gene_name'] = mapped_gene_names[gene_name]
			updated_document['hgnc_gene_id'] = mapped_gene_ids[gene_name]
	
			if columns_dict:
				column_orig_and_new_names = columns_dict
			else:
				column_orig_and_new_names = OrderedDict()
				for column_name in document.keys():
					column_orig_and_new_names[column_name] = column_name

			for original_column_name, new_column_name in column_orig_and_new_names.items():
				updated_document[new_column_name] = document[original_column_name]

			updated_data.append(updated_document)

	csv_reader.data = updated_data

	csv_reader.import_to_db(db.main, collection_name)

	db.main[collection_name].create_index([('hgnc_gene_name', pymongo.ASCENDING)], name='hgnc_gene_name_1')
	db.main[collection_name].create_index([('hgnc_gene_id', pymongo.ASCENDING)], name='hgnc_gene_id_1')


def get_mapped_gene_id_to_gene_name_hgnc_dict(db, gene_list_name, gene_ids, 
	                                          check_for_duplicates=True, store_report=True):
	if check_for_duplicates:
		seen_gene_ids = set()
		duplicate_gene_ids = set()

		for gene_id in gene_ids:
			if gene_id in seen_gene_ids:
				duplicate_gene_ids.add(gene_id)
			else:
				seen_gene_ids.add(gene_id)

		# Remove duplicates from the list
		gene_ids_set = seen_gene_ids - duplicate_gene_ids
	else:
		gene_ids_set = set(gene_ids)

	mapped_gene_ids = {}
	hgnc_genes = db.main.hgnc_genes.find({})
	for hgnc_gene in hgnc_genes:
		gene_id = hgnc_gene['hgnc_id']
		gene_name = hgnc_gene['symbol']

		if gene_id in gene_ids_set:
			mapped_gene_ids[gene_id] = gene_name

	if store_report:
		mapping_report = OrderedDict()
		mapping_report['_id'] = gene_list_name
		if check_for_duplicates:
			mapping_report['duplicate_gene_ids'] = list(duplicate_gene_ids)
			mapping_report['duplicate_gene_ids_num'] = len(duplicate_gene_ids)

		found_gene_ids = list(mapped_gene_ids.keys())
		mapping_report['found_gene_ids'] = found_gene_ids
		mapping_report['found_gene_ids_num'] = len(found_gene_ids)
		not_found_gene_ids = list(gene_ids_set - set(mapped_gene_ids.keys()))
		mapping_report['not_found_gene_ids'] = not_found_gene_ids
		mapping_report['not_found_gene_ids_num'] = len(not_found_gene_ids)
		mapping_report['mapped_gene_ids'] = mapped_gene_ids

		db.main.gene_mapping_reports.delete_one({'_id': gene_list_name})
		db.main.gene_mapping_reports.insert_one(mapping_report)

	return mapped_gene_ids


#########################
### GENERIC GENE LIST ###
#########################

def import_gene_list(db, gene_list_name, file_path):
	'''
	This function is used to import data from text files which contains only gene names 
	'''

	with open(file_path) as f:
		genes = f.readlines()

	# Remove whitespace characters like `\n` at the end of each line (just in case)
	gene_names = [gene.strip() for gene in genes]

	mapped_gene_names, mapped_gene_ids = get_mapped_gene_name_to_hgnc_dicts(db, gene_list_name, gene_names)
	hgnc_gene_ids = list(mapped_gene_ids.values())
	hgnc_gene_ids.sort()

	db.main.gene_lists.delete_one({'_id': gene_list_name})
	db.main.gene_lists.insert_one({'_id': gene_list_name, 'hgnc_gene_ids': hgnc_gene_ids})


################################################
### GDIT: Gene Discovery Informatics Toolkit ###
################################################

def import_gdit_genes(db):
	print('Importing Gene Discovery Informatics Toolkit data...')
	import_gene_collection_with_csv_reader_and_hgnc_data(db, GDIT_CSV, 'gene', 'gdit_genes')


###############
### ENSEMBL ###
###############

def import_ens_proteins(db, build):
	print('Importing ENSEMBL transcript id to protein id...')
	if build == 37:
		source_file = ENS_PROTEINS_37_TSV
		collection_name = 'ens_proteins_37'
	else:
		print('No source file for ENSEMBL Build:', build)	

	columns_dict = OrderedDict()
	columns_dict['Gene stable ID'] = 'gene_id'
	columns_dict['Transcript stable ID'] = 'transcript_id'
	columns_dict['Protein stable ID'] = 'protein_id'
	columns_dict['Chromosome/scaffold name'] = 'chrom'

	import_gene_collection_with_csv_reader_and_hgnc_data(db, source_file, 'HGNC symbol', collection_name, 
		                                                 delimiter='\t', columns_dict=columns_dict,
		                                                 check_for_duplicates=False)

	# Delete transcripts without proteins (e.g. non-coding)
	db.main[collection_name].delete_many({'protein_id': ''})


def get_protein_id_to_hgnc_id_dict(db, build):
	if build == 37:
		ens_proteins_collection = 'ens_proteins_37'
	else:
		print('No source file for ENSEMBL Build:', build)	

	ens_protein_id_to_hgnc_id = {}
	ens_proteins = db.main[ens_proteins_collection].find({})

	for ens_protein in ens_proteins:
		ens_protein_id_to_hgnc_id[ens_protein['protein_id']] = ens_protein['hgnc_gene_id']

	return ens_protein_id_to_hgnc_id


##############
### STRING ###
##############

def import_string(db, version):
	if version == 10:
		print('Importing STRING v10 PPIs...')
		source_file = STRING_V10_TXT
		ens_protein_id_to_hgnc_id = get_protein_id_to_hgnc_id_dict(db, 37)
	else:
		print('No source file for STRING version:', version)
	
	collection_name = 'string_v' + str(version)
	
	db.main[collection_name].drop()

	input_file = open(source_file, 'rt')
	reader = csv.reader(input_file, delimiter=' ')

	string_data = []

	headers = next(reader)
	total_lines = len(open(source_file).readlines())
	line_number = 0
	bar = ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()
	for row in reader:
		line_number += 1

		version_1, protein_1 = row[0].split('.')
		version_2, protein_2 = row[1].split('.')

		# Skip protein interactions which cannot be linked to the gene hgnc ids
		if (protein_1 not in ens_protein_id_to_hgnc_id or
			protein_2 not in ens_protein_id_to_hgnc_id):
			continue

		ppi = OrderedDict()
		ppi['protein_1'] = protein_1
		ppi['protein_2'] = protein_2
		ppi['hgnc_gene_id_1'] = ens_protein_id_to_hgnc_id[protein_1]
		ppi['hgnc_gene_id_2'] = ens_protein_id_to_hgnc_id[protein_2]
		ppi['neighborhood'] = int(row[2])
		ppi['fusion'] = int(row[3])
		ppi['cooccurence'] = int(row[4])
		ppi['coexpression'] = int(row[5])
		ppi['experimental'] = int(row[6])
		ppi['database'] = int(row[7])
		ppi['textmining'] = int(row[8])
		ppi['combined_score'] = int(row[9])

		if line_number % 1000 == 0:
			db.main[collection_name].insert_many(string_data)
			string_data = []
		else:
			string_data.append(ppi)
		
		bar.update((line_number + 0.0) / total_lines)
	bar.finish()

	if len(string_data) > 0:
		db.main[collection_name].insert_many(string_data)
		
	db.main[collection_name].create_index([('hgnc_gene_id_1', pymongo.ASCENDING)], name='hgnc_gene_id_1_1')
	db.main[collection_name].create_index([('hgnc_gene_id_2', pymongo.ASCENDING)], name='hgnc_gene_id_2_1')


############################
### GeVIR, LOEUF, VIRLoF ###
############################

def import_gevir_loeuf_virlof_scores(db):
	print('Importing GeVIR, LOEUF and VIRLoF scores...')

	columns_dict = OrderedDict()
	columns_dict['gevir_percentile'] = 'gevir'
	columns_dict['loeuf_percentile'] = 'loeuf'
	columns_dict['virlof_percentile'] = 'virlof'

	import_gene_collection_with_csv_reader_and_hgnc_data(db, GEVIR_LOEUF_VIRLOF_SCORES_CSV, 'gnomad_gene_name', 
	                                                     'gevir_genes', columns_dict=columns_dict)


#################
### UNEECON-G ###
#################

def import_uneecon_genes(db):
	print('Importing UNEECON scores...')
	uneecon_g = CsvReader(UNEECON_G_TSV, delimiter='\t')
	gene_names = []

	for document in uneecon_g.data:
		gene_info = document['#gene']
		gene_id, gene_name = gene_info.split('|')
		document['gene_name'] = gene_name

	columns_dict = OrderedDict()
	columns_dict['UNEECON-G'] = 'score'

	import_gene_collection_with_csv_reader_and_hgnc_data(db, UNEECON_G_TSV, 'gene_name', 'uneecon_genes', 
	                                                     columns_dict=columns_dict, csv_reader=uneecon_g)


##############
### DOMINO ###
##############

def get_domino_mapped_gene_names(db, file_path, gene_list_name, gene_name_column_name):
	domino = CsvReader(file_path, delimiter='\t')
	gene_names = []
	for document in domino.data:
		gene_names.append(document[gene_name_column_name])

	mapped_gene_names, mapped_gene_ids = get_mapped_gene_name_to_hgnc_dicts(db, gene_list_name, gene_names)	
	return mapped_gene_names, mapped_gene_ids


def get_gene_name_to_inheritance_dict_from_domino_tsv(db, file_path, gene_list_name):
	mapped_gene_names, mapped_gene_ids = get_domino_mapped_gene_names(db, file_path, gene_list_name, 'gene_name')
	gene_name_to_inheritance = {}

	domino = CsvReader(file_path, delimiter='\t')

	for document in domino.data:
		gene_name = document['gene_name']
		if gene_name in mapped_gene_names:
			gene_name = mapped_gene_names[gene_name]
			inheritance = document['inheritance']
			if inheritance == 'AD' or inheritance == 'AR':
				gene_name_to_inheritance[gene_name] = inheritance

	return gene_name_to_inheritance


def import_domino_genes(db):
	'''
	Some gene names in DOMINO train/validation datasets are outdated in comparison with DOMINO final dataset.
	To merge these datasets, we've updated gene names in each of them separately, consequently
	there are 3 mapping reports in "gene_mapping_reports" collection

	GIF (up-to-date name: CBLIF) is present in the train dataset, but is NOT present in the final gene scores list
	consequently we also did not use this gene for the training. 
	'''
	print('Importing DOMINO scores...')

	domino_train = get_gene_name_to_inheritance_dict_from_domino_tsv(db, DOMINO_TRAIN_GENES_TSV, 'domino_train')
	domino_validation = get_gene_name_to_inheritance_dict_from_domino_tsv(db, DOMINO_VALIDATION_GENES_TSV, 'domino_validation')
	final_mapped_gene_names, final_mapped_gene_ids = get_domino_mapped_gene_names(db, DOMINO_FINAL_GENE_SCORES_TSV, 
		                                                                          'domino_final', '#HGNC ID')
	domino_final = CsvReader(DOMINO_FINAL_GENE_SCORES_TSV, delimiter='\t')

	for document in domino_final.data:
		gene_name = document['#HGNC ID']

		if gene_name not in final_mapped_gene_names:
			continue
		else:
			gene_name = final_mapped_gene_names[gene_name]

		if gene_name in domino_train:
			document['inheritance'] = domino_train[gene_name]
			document['train'] = True
			document['validation'] = False
		elif gene_name in domino_validation:
			document['inheritance'] = domino_validation[gene_name]
			document['train'] = False
			document['validation'] = True
		else:
			document['inheritance'] = ''
			document['train'] = False
			document['validation'] = False

	columns_dict = OrderedDict()
	columns_dict['inheritance'] = 'inheritance'
	columns_dict['train'] = 'train'
	columns_dict['validation'] = 'validation'

	columns_dict['Score'] = 'score'
	columns_dict['STRING-combined score'] = 'string_combined_score'
	columns_dict['STRING-experimental score'] = 'string_experimental_score'
	columns_dict['STRING-textmining score'] = 'string_textmining_score'
	columns_dict['ExAC-pRec'] = 'exac_prec'
	columns_dict['ExAC-missense z-score'] = 'exac_missense_z_score'
	columns_dict["PhyloP at 5'-UTR"] = 'phylo_pat_5_prime_utr'
	columns_dict['Number donor/number synonymous'] = 'n_donor_divided_by_n_synonymous'
	columns_dict['mRNA half-life->10h'] = 'mrna_half_life_gt_10h'
	columns_dict['lda score'] = 'lda_score'

	import_gene_collection_with_csv_reader_and_hgnc_data(db, DOMINO_FINAL_GENE_SCORES_TSV, '#HGNC ID', 'domino_genes', 
	                                                     columns_dict=columns_dict, csv_reader=domino_final, store_report=False)


################################################################
### Map HGNC gene names/ids to GnomAD canonical transcripts ###
################################################################

def map_gnomad_hgnc_ids_to_canonical_transcripts(db):
	gene_names = []
	gene_name_to_transcript_id = {}
	gnomad_genes = db.exac.genes.find({ "canonical_transcript": { "$exists": True } })
	for gnomad_gene in gnomad_genes:
		gene_name = gnomad_gene['gene_name']
		gene_names.append(gnomad_gene['gene_name'])
		gene_name_to_transcript_id[gene_name] = gnomad_gene['canonical_transcript']

	mapped_gene_names, mapped_gene_ids = get_mapped_gene_name_to_hgnc_dicts(db, 'gnomad_canonical_transcripts', gene_names)	

	hgnc_id_to_canonical_transcript = {}
	for gene_name, hgnc_gene_id in mapped_gene_ids.items():
		hgnc_id_to_canonical_transcript[hgnc_gene_id] = gene_name_to_transcript_id[gene_name]

	db.main.mapping_dicts.delete_one({'_id': 'hgnc_id_to_canonical_transcript'})
	db.main.mapping_dicts.insert_one({'_id': 'hgnc_id_to_canonical_transcript', 'mapping_dict': hgnc_id_to_canonical_transcript})


#########################################
### GnomAD v2.1.1 Structural Variants ###
#########################################

def import_gnomad_svs(db):
	gnomad_svs = CsvReader(GNOMAD_CONTROL_ONLY_SVS, delimiter='\t')

	gene_names = set()
	for document in gnomad_svs.data:
		lof_genes =  document['PROTEIN_CODING__LOF']
		if lof_genes != 'NA':
			gene_names |= set(lof_genes.split(','))

	mapped_gene_names, mapped_gene_ids = get_mapped_gene_name_to_hgnc_dicts(db, 'gnomad_lof_svs', gene_names, 
	                                                                        check_for_duplicates=False)

	for document in gnomad_svs.data:
		lof_hgnc_gene_ids = []
		lof_genes =  document['PROTEIN_CODING__LOF']
		if lof_genes != 'NA':
			gene_names = lof_genes.split(',')
			for gene_name in gene_names:
				if gene_name in mapped_gene_ids:
					lof_hgnc_gene_ids.append(mapped_gene_ids[gene_name])
		document['lof_hgnc_gene_ids'] = lof_hgnc_gene_ids
		document.move_to_end('lof_hgnc_gene_ids', last=False)

	gnomad_svs.import_to_db(db.main, 'gnomad_control_only_svs')


###########
### GPP ###
###########

def import_gpp_genes(db):
	print('Importing GPP scores...')
	columns_dict = OrderedDict()
	columns_dict['score'] = 'score'

	import_gene_collection_with_csv_reader_and_hgnc_data(db, GPP_GENE_SCORES_CSV, 'gene_name', 
	                                                     'gpp_genes', columns_dict=columns_dict)

def import_gdp_genes(db):
	print('Importing GDP scores...')
	columns_dict = OrderedDict()
	columns_dict['score'] = 'score'

	import_gene_collection_with_csv_reader_and_hgnc_data(db, GDP_GENE_SCORES_CSV, 'gene_name', 
	                                                     'gdp_genes', columns_dict=columns_dict)


###################
### GENE4DENOVO ###
###################

def import_gene4denovo(db):
	print('Importing Gene4Denovo genes...')
	import_gene_collection_with_csv_reader_and_hgnc_data(db, GENE4DENOVO_TSV, 'Gene', 'gene4denovo', 
	                                                     delimiter='\t', check_for_duplicates=False)


def main():
	db = MongoDB()

	# HGNC
	#import_hgnc_genes(db)

	# GDIT
	#import_gdit_genes(db)

	# ENSEMBL
	#import_ens_proteins(db, 37)

	# STRING
	#import_string(db, 10)

	# GeVIR, LOEUF, VIRLoF
	#import_gevir_loeuf_virlof_scores(db)

	# UNEECON
	#import_uneecon_genes(db)

	# DOMINO
	#import_domino_genes(db)

	# HGNC to gnomAD canonical transcript, adds document to 'mapping' collection
	#map_gnomad_hgnc_ids_to_canonical_transcripts(db)

	# GnomAD Structural Variants 
	#import_gnomad_svs(db)

	# GPP
	#import_gpp_genes(db)
	#import_gdp_genes(db)
	#import_gene_list(db, 'gpp_train_disease', GPP_TRAIN_DISEASE_LIST)
	#import_gene_list(db, 'gpp_train_tolerant', GPP_TRAIN_TOLERANT_LIST)

	# Gene4Denovo
	#import_gene4denovo(db)

	# Various gene Lists
	#import_gene_list(db, 'olfactory', OLFACTORY_GENE_LIST)
	#import_gene_list(db, 'cell_essential', CELL_ESSENTIAL_GENE_LIST)
	#import_gene_list(db, 'cell_non_essential', CELL_NON_ESSENTIAL_GENE_LIST)
	#import_gene_list(db, 'severe_hi', SEVERE_HI_GENE_LIST)



if __name__ == "__main__":
	sys.exit(main())