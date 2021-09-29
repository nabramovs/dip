import os
import sys
from collections import OrderedDict
from common import MongoDB

AD_LABEL_CODE = 0
AR_LABEL_CODE = 1

TOLERANT_LABEL_CODE = 0
DISEASE_LABEL_CODE = 1

class GeneGroups():
	def __init__(self, db, valid_gene_ids=set()):
		self.db = db
		if valid_gene_ids:
			self.valid_gene_ids = set(valid_gene_ids)
		else:
			self.valid_gene_ids = set(get_gene_list(db, 'domino_gene_ids'))

		self.gene_id_to_gene_name = get_mapping_dict(db, 'hgnc_id_to_gene_name')

		# DOMINO Train
		self.domino_train_ad_gene_ids = get_gene_list(db, 'domino_train_ad_gene_ids', valid_gene_ids=self.valid_gene_ids)
		self.domino_train_ar_gene_ids = get_gene_list(db, 'domino_train_ar_gene_ids', valid_gene_ids=self.valid_gene_ids)

		domino_train_dict = {AD_LABEL_CODE: self.domino_train_ad_gene_ids, AR_LABEL_CODE: self.domino_train_ar_gene_ids}
		self.domino_train_gene_ids, self.domino_train_labels = \
		self._convert_group_id_to_gene_list_dict_to_ml_x_and_y_lists(domino_train_dict)

		# DOMINO Validation
		self.domino_validation_ad_gene_ids = get_gene_list(db, 'domino_validation_ad_gene_ids', valid_gene_ids=self.valid_gene_ids)
		self.domino_validation_ar_gene_ids = get_gene_list(db, 'domino_validation_ar_gene_ids', valid_gene_ids=self.valid_gene_ids)

		domino_validation_dict = {AD_LABEL_CODE: self.domino_validation_ad_gene_ids, AR_LABEL_CODE: self.domino_validation_ar_gene_ids}
		self.domino_validation_gene_ids, self.domino_validation_labels = \
		self._convert_group_id_to_gene_list_dict_to_ml_x_and_y_lists(domino_validation_dict)

		# GPP Train
		self.gpp_train_tolerant_gene_ids = get_gene_list(db, 'gpp_train_tolerant', valid_gene_ids=self.valid_gene_ids)
		self.gpp_train_disease_gene_ids = get_gene_list(db, 'gpp_train_disease', valid_gene_ids=self.valid_gene_ids)

		gpp_train_dict = {TOLERANT_LABEL_CODE: self.gpp_train_tolerant_gene_ids, DISEASE_LABEL_CODE: self.gpp_train_disease_gene_ids}
		self.gpp_train_gene_ids, self.gpp_train_labels = \
		self._convert_group_id_to_gene_list_dict_to_ml_x_and_y_lists(gpp_train_dict)

		# GDIT All
		self.gdit_ad_gene_ids = get_gene_list(db, 'gdit_ad', valid_gene_ids=self.valid_gene_ids)
		self.gdit_ar_gene_ids = get_gene_list(db, 'gdit_ar', valid_gene_ids=self.valid_gene_ids)
		self.gdit_ad_ar_gene_ids = get_gene_list(db, 'gdit_ad_ar', valid_gene_ids=self.valid_gene_ids)
		self.gdit_omim_gene_ids = get_gene_list(db, 'gdit_omim', valid_gene_ids=self.valid_gene_ids)
		
		self.gdit_lethal_b_gene_ids = get_gene_list(db, 'gdit_lethal_b', valid_gene_ids=self.valid_gene_ids)
		self.gdit_lethal_a_gene_ids = get_gene_list(db, 'gdit_lethal_a', valid_gene_ids=self.valid_gene_ids)

		self.gdit_lethal_ad_gene_ids = gene_set_to_sorted_list(set(self.gdit_ad_gene_ids) & set(self.gdit_lethal_a_gene_ids))
		self.gdit_lethal_ar_gene_ids = gene_set_to_sorted_list(set(self.gdit_ar_gene_ids) & set(self.gdit_lethal_a_gene_ids))
		self.gdit_lethal_ad_ar_gene_ids = gene_set_to_sorted_list(set(self.gdit_ad_ar_gene_ids) & set(self.gdit_lethal_a_gene_ids))
		self.gdit_lethal_omim_gene_ids = gene_set_to_sorted_list(set(self.gdit_omim_gene_ids) & set(self.gdit_lethal_a_gene_ids))
		
		# Olfactory
		self.olfactory_gene_ids = get_gene_list(db, 'olfactory', valid_gene_ids=self.valid_gene_ids)

		# Cell Essential/Non-essential
		self.cell_essential_gene_ids = get_gene_list(db, 'cell_essential', valid_gene_ids=self.valid_gene_ids)
		self.cell_non_essential_gene_ids = get_gene_list(db, 'cell_non_essential', valid_gene_ids=self.valid_gene_ids)

		# Severe HI genes
		self.severe_hi_gene_ids = get_gene_list(db, 'severe_hi', valid_gene_ids=self.valid_gene_ids)

		# Gene4Denovo
		self.gene4denovo_gene_ids = get_gene_list(db, 'gene4denovo', valid_gene_ids=self.valid_gene_ids)

		# Mouse Het Lethal
		self.gdit_mouse_het_lethal_gene_ids = get_gene_list(db, 'gdit_mouse_het_lethal', valid_gene_ids=self.valid_gene_ids)
		self.gdit_mouse_hom_lethal_gene_ids = get_gene_list(db, 'gdit_mouse_hom_lethal', valid_gene_ids=self.valid_gene_ids)


	def _convert_group_id_to_gene_list_dict_to_ml_x_and_y_lists(self, group_id_to_gene_list):
		gene_id_to_group_dict = OrderedDict()
		for group_id, gene_ids in group_id_to_gene_list.items():
			for gene_id in gene_ids:
				gene_id_to_group_dict[gene_id] = group_id
		X = list(gene_id_to_group_dict.keys())
		X.sort()
		y = []
		for gene_id in X:
			y.append(gene_id_to_group_dict[gene_id])
		return X, y


############################
### Gene Lists and Dicts ###
############################

def gene_set_to_sorted_list(gene_set):
	gene_list = list(gene_set)
	gene_list.sort()
	return gene_list

def create_hgnc_id_to_gene_name_dict(db):
	hgnc_id_to_gene_name = {}
	hgnc_genes = db.main.hgnc_genes.find({})
	for hgnc_gene in hgnc_genes:
		hgnc_id_to_gene_name[hgnc_gene['hgnc_id']] = hgnc_gene['symbol']

	db.main.mapping_dicts.delete_one({'_id': 'hgnc_id_to_gene_name'})
	db.main.mapping_dicts.insert_one({'_id': 'hgnc_id_to_gene_name', 'mapping_dict': hgnc_id_to_gene_name})


def get_mapping_dict(db, dict_name):
	mapping_dict = db.main.mapping_dicts.find_one({'_id': dict_name})
	return mapping_dict['mapping_dict']


def add_gene_list(db, gene_list_name, gene_ids):
	gene_ids.sort()

	db.main.gene_lists.delete_one({'_id': gene_list_name})
	db.main.gene_lists.insert_one({'_id': gene_list_name, 'hgnc_gene_ids': gene_ids})


def get_gene_list(db, gene_list_name, valid_gene_ids=set()):
	gene_list = db.main.gene_lists.find_one({'_id': gene_list_name})
	gene_ids = gene_list['hgnc_gene_ids']
	
	if valid_gene_ids:
		gene_ids = set(gene_ids) & set(valid_gene_ids)
		gene_ids = list(gene_ids)

	gene_ids.sort()
	return gene_ids


def create_domino_gene_lists(db):
	domino_gene_ids = []

	train_ad_gene_ids = []
	train_ar_gene_ids = []

	validation_ad_gene_ids = []
	validation_ar_gene_ids = []

	domino_genes = db.main.domino_genes.find({})

	for domino_gene in domino_genes:
		gene_id = domino_gene['hgnc_gene_id']
		inheritance = domino_gene['inheritance']
		domino_gene_ids.append(gene_id)

		if domino_gene['train']:
			if inheritance == 'AD':
				train_ad_gene_ids.append(gene_id)
			elif inheritance == 'AR':
				train_ar_gene_ids.append(gene_id)
		elif domino_gene['validation']:
			if inheritance == 'AD':
				validation_ad_gene_ids.append(gene_id)
			elif inheritance == 'AR':
				validation_ar_gene_ids.append(gene_id)

	add_gene_list(db, 'domino_gene_ids', domino_gene_ids)
	add_gene_list(db, 'domino_train_ad_gene_ids', train_ad_gene_ids)
	add_gene_list(db, 'domino_train_ar_gene_ids', train_ar_gene_ids)
	add_gene_list(db, 'domino_validation_ad_gene_ids', validation_ad_gene_ids)
	add_gene_list(db, 'domino_validation_ar_gene_ids', validation_ar_gene_ids)


def create_gdit_gene_lists(db):
	domino_train_and_validation_gene_ids = set()
	domino_genes = db.main.domino_genes.find({ "$or": [ { "train": True }, { "validation": True } ] })
	for domino_gene in domino_genes:
		domino_train_and_validation_gene_ids.add(domino_gene['hgnc_gene_id'])

	gpp_train_gene_ids = set(get_gene_list(db, 'gpp_train_disease') + get_gene_list(db, 'gpp_train_tolerant'))

	ad_gene_ids = set()
	ar_gene_ids = set()
	ad_ar_gene_ids = set()
	omim_gene_ids = set()

	lethal_a = set()
	lethal_b = set()

	mouse_het_lethal = set()
	mouse_hom_lethal = set()

	gdit_genes = db.main.gdit_genes.find({})

	for gdit_gene in gdit_genes:
		gene_id = gdit_gene['hgnc_gene_id']
		inheritance = gdit_gene['Inheritance_pattern']

		if inheritance == 'AD':
			ad_gene_ids.add(gene_id)
		elif inheritance == 'AR':
			ar_gene_ids.add(gene_id)
		elif inheritance == 'AR,AD':
			ad_ar_gene_ids.add(gene_id)

		if gdit_gene['omim'] == 'Y':
			omim_gene_ids.add(gene_id)

		if gdit_gene['human_lethal_B'] == 'Y':
			lethal_b.add(gene_id)

		if gdit_gene['human_lethal_A'] == 'Y':
			lethal_a.add(gene_id)

		if gdit_gene['lethal_het_mouse'] == 'Y':
			mouse_het_lethal.add(gene_id)

		if gdit_gene['lethal_mouse'] == 'Y':
			mouse_hom_lethal.add(gene_id)

	gdit_lists = OrderedDict()
	gdit_lists['gdit_ad'] = list(ad_gene_ids)
	gdit_lists['gdit_ar'] = list(ar_gene_ids)
	gdit_lists['gdit_ad_ar'] = list(ad_ar_gene_ids)
	gdit_lists['gdit_no_domino_ad'] = list(ad_gene_ids - domino_train_and_validation_gene_ids)
	gdit_lists['gdit_no_domino_ar'] = list(ar_gene_ids - domino_train_and_validation_gene_ids)
	gdit_lists['gdit_no_domino_ad_ar'] = list(ad_ar_gene_ids - domino_train_and_validation_gene_ids)
	gdit_lists['gdit_no_gpp_ad'] = list(ad_gene_ids - gpp_train_gene_ids)
	gdit_lists['gdit_no_gpp_ar'] = list(ar_gene_ids - gpp_train_gene_ids)
	gdit_lists['gdit_no_gpp_ad_ar'] = list(ad_ar_gene_ids - gpp_train_gene_ids)
	gdit_lists['gdit_omim'] = list(omim_gene_ids)
	gdit_lists['gdit_lethal_b'] = list(lethal_b)
	gdit_lists['gdit_lethal_a'] = list(lethal_a)
	gdit_lists['gdit_mouse_het_lethal'] = list(mouse_het_lethal)
	gdit_lists['gdit_mouse_hom_lethal'] = list(mouse_hom_lethal)

	for gdit_list_name, gdit_list in gdit_lists.items():
		add_gene_list(db, gdit_list_name, gdit_list)


def create_gene4denovo_gene_list(db):
	denovo_genes = db.main.gene4denovo.find({ "FDR": { "$lte": "0.05" },
											  "$or": [ { "Groups": "ASD" }, { "Groups": "EE" }, { "Groups": "ID" }] }) #, { "Groups": "UDD" }] }) #] })
											  #"$or": [ { "Groups": "UDD" }] })
	denovo_gene_ids = set()

	for denovo_gene in denovo_genes:
		denovo_gene_ids.add(denovo_gene['hgnc_gene_id'])

	denovo_gene_ids = list(denovo_gene_ids)
	add_gene_list(db, 'gene4denovo', denovo_gene_ids)


#####################################
### GENERAL METHODS ON GENE LISTS ###
#####################################

def get_common_domino_gpp_and_gevir_gene_ids(db, no_domino_train=False, no_domino_validation=True, no_gpp_train=False, include_uneecon=False):
	gpp_gene_ids = set()
	gpp_genes = db.main.gpp_genes.find({})
	for gpp_gene in gpp_genes:
		gpp_gene_ids.add(gpp_gene['hgnc_gene_id'])

	domino_gene_ids = set()
	domino_genes = db.main.domino_genes.find({})
	for domino_gene in domino_genes:
		domino_gene_ids.add(domino_gene['hgnc_gene_id'])

	gevir_gene_ids = set()
	gevir_genes = db.main.gevir_genes.find({})
	for gevir_gene in gevir_genes:
		gevir_gene_ids.add(gevir_gene['hgnc_gene_id'])

	common_gene_ids = gpp_gene_ids & domino_gene_ids & gevir_gene_ids

	if include_uneecon:
		uneecon_gene_ids = set()
		uneecon_genes = db.main.uneecon_genes.find({})
		for uneecon_gene in uneecon_genes:
			uneecon_gene_ids.add(uneecon_gene['hgnc_gene_id'])
		common_gene_ids = common_gene_ids & uneecon_gene_ids	

	if no_domino_train:
		common_gene_ids -= set(get_gene_list(db, 'domino_train_ad_gene_ids'))
		common_gene_ids -= set(get_gene_list(db, 'domino_train_ar_gene_ids'))

	if no_gpp_train:
		common_gene_ids -= set(get_gene_list(db, 'gpp_train_tolerant'))
		common_gene_ids -= set(get_gene_list(db, 'gpp_train_disease'))

	if no_domino_validation:
		common_gene_ids -= set(get_gene_list(db, 'domino_validation_ad_gene_ids'))
		common_gene_ids -= set(get_gene_list(db, 'domino_validation_ar_gene_ids'))		

	return common_gene_ids


def get_gene_id_to_virlof(db):
	gene_id_to_virlof = {}
	gevir_genes = db.main.gevir_genes.find({})

	for gevir_gene in gevir_genes:
		gene_id_to_virlof[gevir_gene['hgnc_gene_id']] = gevir_gene['virlof']
	return gene_id_to_virlof


def get_gene_id_to_string_ppi_num(db, ppi_type):
	gene_id_to_string_ppi_num = {}
	string_genes = db.main.string_genes_v10.find({})

	for string_gene in string_genes:
		gene_id_to_string_ppi_num[string_gene['hgnc_gene_id']] = len(string_gene[ppi_type])
	return gene_id_to_string_ppi_num


def main():
	db = MongoDB()

	# Uncomment to create the gene list 
	# Create gene group lists/mapping dicts
	#create_hgnc_id_to_gene_name_dict(db)
	#create_domino_gene_lists(db)
	#create_gdit_gene_lists(db)
	#create_gene4denovo_gene_list(db)


	'''
	# Testing
	#gg = GeneGroups(db)
	valid_gene_ids = set(get_gene_list(db, 'domino_gene_ids'))
	gpp_train_tolerant_gene_ids = get_gene_list(db, 'gpp_train_tolerant', valid_gene_ids=valid_gene_ids)
	gpp_train_disease_gene_ids = get_gene_list(db, 'gpp_train_disease', valid_gene_ids=valid_gene_ids)	

	valid_gene_ids = valid_gene_ids - set(gpp_train_tolerant_gene_ids) - set(gpp_train_disease_gene_ids)
	
	gg = GeneGroups(db, valid_gene_ids=valid_gene_ids)

	print('AD', len(gg.gdit_ad_gene_ids))
	print('AR', len(gg.gdit_ar_gene_ids))
	print('AD,AR', len(gg.gdit_ad_ar_gene_ids))
	print('OLF', len(gg.olfactory_gene_ids))
	print('CE', len(gg.cell_essential_gene_ids))
	print('CNE', len(gg.cell_non_essential_gene_ids))
	'''


if __name__ == "__main__":
	sys.exit(main())