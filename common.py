import csv
import pymongo
import numpy as np
import math
from collections import OrderedDict
from decimal import Decimal
from scipy.stats import fisher_exact

#################
### CONSTANTS ###
#################

DB_HOST = 'localhost'
DB_PORT = 27017
DB_NAME_GR = 'gr' 
DB_NAME_EXAC = 'exac'
			
			
class MongoDB():
	"""Database Client."""
	def __init__(self):
		client = pymongo.MongoClient(host=DB_HOST, port=DB_PORT, document_class=OrderedDict)
		self.main = client[DB_NAME_GR]
		self.exac = client[DB_NAME_EXAC]



def file_len(fname):
	"""Calculate length of a file."""
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1


def is_float(x):
	"""Check if value (e.g. string) can be converted to float."""
	try:
		a = float(x)
	except ValueError:
		return False
	else:
		return True


def is_int(x):
	"""Check if value (e.g. string) can be converted to integer."""
	try:
		a = float(x)
		b = int(a)
	except ValueError:
		return False
	else:
		return a == b


def calculate_percentiles(ranked_list, reverse=False):
	"""Return list of percetiles based on number of elements in input list."""
	percentiles = OrderedDict()
	max_num = len(ranked_list)

	percentile = 0.0
	for x in range(0, max_num):
		if reverse:
			percentile = (1 - float(x + 1) / max_num) * 100
		else:
			percentile = float(x + 1) / max_num * 100
		percentiles[ranked_list[x]] = percentile

	return percentiles


def write_table_to_csv(table, output_csv, delimiter=','):
	"""Write table (list of lists) to csv."""
	output_file = open(output_csv,'w+')
	writer = csv.writer(output_file, delimiter=delimiter)

	for row in table:
		writer.writerow(row)

	output_file.close()


def sort_dict_by_values(dictionary, reverse=False):
	"""Return dictionary sorted by values."""
	sorted_tuples = sorted(dictionary.items(), key=lambda x: x[1], reverse=reverse)
	result = OrderedDict()
	for x in range(0, len(sorted_tuples)):
		result[sorted_tuples[x][0]] = sorted_tuples[x][1]
	return result


def float_to_sci_str(num):
	return "{:.2E}".format(Decimal(num))


def proportion_to_percents_str(proportion):
	return "{0:.1f}".format(proportion*100)


def get_sorted_gene_list(db, collection_name, score_field, reverse=False, ignored_genes=set(), filters={}):
	gene_scores = {}
	genes = db.drt[collection_name].find(filters)
	for gene in genes:
		gene_id = gene['hgnc_id']
		if gene_id not in ignored_genes:
			gene_scores[gene['hgnc_id']] = gene[score_field]

	gene_scores = sort_dict_by_values(gene_scores, reverse=reverse)
	return gene_scores


def remove_non_valid_keys_from_dict(dictionary, valid_keys):
	updated_dict = OrderedDict()
	for key, value in dictionary.iteritems():
		if key in valid_keys:
			updated_dict[key] = value
	return updated_dict


def get_str_ratio(n, total, only_ratio=False):
	ratio = float(n * 100 / total)
	if only_ratio:
		str_ratio = "{:.2f}".format(ratio)
	else:
		str_ratio = "{} ({:.2f}%)".format(n, ratio)
	return str_ratio


def report_gene_group_enrichment_in_the_subset(group_name, gene_subset_ids, all_gene_ids, gene_group_ids):
	gene_subset_ids = set(gene_subset_ids)
	all_gene_ids = set(all_gene_ids)
	gene_group_ids = set(gene_group_ids)

	subset = len(gene_subset_ids)
	total = len(all_gene_ids)
	group_and_all = len(gene_group_ids & all_gene_ids)
	group_and_subset = len(gene_group_ids & gene_subset_ids)

	fe, p = fisher_exact([[group_and_subset, subset],
						  [group_and_all, total]])
	print([[group_and_subset, subset],
						  [group_and_all, total]])
	print('### {} ###'.format(group_name, group_and_all))
	print('Examined subset {}/{}, {:.2f}%'.format(subset, total, subset*100/total))
	print('{} in the subset {}/{}, {:.2f}%'.format(group_name, group_and_subset,
										           group_and_all, group_and_subset*100/group_and_all))
	print('FE: {:.3f}, P-value: {}'.format(fe, p))


def get_metric_ranked_gene_scores(db, collection_name, score_field, reverse=False, valid_gene_ids=set()):
	metric_scores = OrderedDict()
	metric_genes = db.main[collection_name].find({})
	for metric_gene in metric_genes:
		gene_id = metric_gene['hgnc_gene_id']
		if valid_gene_ids and gene_id not in valid_gene_ids:
			continue		
		metric_scores[gene_id] = metric_gene[score_field]
	metric_scores = sort_dict_by_values(metric_scores, reverse=reverse)
	return metric_scores
	

# Modified J.Vo answer from here:
# https://stackoverflow.com/questions/30098263/inserting-a-document-with-pymongo-invaliddocument-cannot-encode-object
def correct_encoding(obj):
	"""Correct the encoding of python dictionaries so they can be encoded to mongodb
	inputs
	-------
	dictionary : dictionary instance to add as document
	output
	-------
	new : new dictionary with (hopefully) corrected encodings"""

	if isinstance(obj, dict):
		new_dict = {}
		for key, val in obj.items():
			val = correct_encoding(val)
			new_dict[key] = val
		return new_dict
	elif isinstance(obj, list):
		new_list = []
		for val in obj:
			val = correct_encoding(val)
			new_list.append(val)
		return new_list
	else:
		if isinstance(obj, np.bool_):
			obj = bool(obj)

		if isinstance(obj, np.int64):
			obj = int(obj)

		if isinstance(obj, np.float64):
			obj = float(obj)
		return obj


def get_keys_from_dict_based_on_value_threshold(dictionary, threshold, comparison_mode):
	keys = []
	for key, value in dictionary.items():
		if comparison_mode == '>=' and value >= threshold:
			keys.append(key)
		elif comparison_mode == '<=' and value <= threshold:
			keys.append(key)
		elif comparison_mode == '>' and value > threshold:
			keys.append(key)
		elif comparison_mode == '<' and value < threshold:
			keys.append(key)
		elif comparison_mode == '==' and value == threshold:
			keys.append(key)
	return keys


def calculate_clf_performance(tp, fp, p_all):
	prec = tp / (tp + fp)
	rec = tp / p_all
	f1 = 2 * (prec * rec) / (prec + rec)

	metrics = OrderedDict()
	metrics['Precision'] = '{:.2f}%'.format(prec * 100)
	metrics['Recall'] = '{:.2f}%'.format(rec * 100)
	metrics['F1'] = '{:.2f}%'.format(f1 * 100)
	return metrics