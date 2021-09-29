import os
import sys
import progressbar
import pymongo
import numpy as np

from pathlib import Path
from scipy.stats import fisher_exact, mannwhitneyu

from features import Features
from progressbar import ProgressBar, ETA, Percentage, Bar
from collections import OrderedDict
from common import MongoDB, sort_dict_by_values, calculate_percentiles, get_str_ratio
from common import write_table_to_csv, report_gene_group_enrichment_in_the_subset
from common import get_metric_ranked_gene_scores
from common import get_keys_from_dict_based_on_value_threshold
from common import calculate_clf_performance
from gene_groups import GeneGroups, get_common_domino_gpp_and_gevir_gene_ids
from gene_groups import get_gene_id_to_virlof, get_gene_id_to_string_ppi_num
from models import get_ranked_gene_enrichment_stats
from models import train_cv_d_rt_model, test_d_rt_model
from models import export_gene_metrics
from feature_selection import get_model_clf_and_feature_names, get_all_feature_names

from string_ppi import analyse_domino_direct_ad_method_performance
from string_ppi import analyse_string_knn_method_performance

PROGRESS_BAR_WIDGETS = [Percentage(), ' ', Bar(), ' ', ETA()]

TABLE_FOLDER = Path('tables/')

DRT_NAME = 'DIP'
DR_T_NAME = 'DND'
D_RT_NAME = 'ADR'

def get_gene_ids(db, collection_name):
	gene_ids = []
	genes = db.main[collection_name].find({})
	for gene in genes:
		gene_ids.append(gene['hgnc_gene_id'])
	return gene_ids


def get_evaluation_common_gene_ids(db):
	dip_genes = set(get_gene_ids(db, 'dip_genes'))
	domino_genes = set(get_gene_ids(db, 'domino_genes'))
	gpp_genes = set(get_gene_ids(db, 'gpp_genes'))
	virlof_genes = set(get_gene_ids(db, 'gevir_genes'))
	common_gene_ids = dip_genes & domino_genes & gpp_genes & virlof_genes
	return common_gene_ids


def get_gene_scores(db, collection_name, score_key, valid_gene_ids=set(), reverse=False):
	'''
	GPP: score_key='score', reverse='True'
	DOMINO: score_key='score', reverse='True'
	DIP: score_key='dip_rank', reverse='False'
	VIRLoF: score_key='virlof', reverse='False' 
	'''
	gene_scores = {}
	genes = db.main[collection_name].find({})
	for gene in genes:
		gene_id = gene['hgnc_gene_id']
		if gene_id in valid_gene_ids:
			gene_scores[gene_id] = gene[score_key]
	gene_scores = sort_dict_by_values(gene_scores, reverse=reverse)
	return gene_scores


############################
### METHODS GENES REPORT ###
############################

def report_methods_genes(db):
	domino_genes = get_gene_ids(db, 'domino_genes')
	gevir_genes = get_gene_ids(db, 'gevir_genes')
	gpp_genes = get_gene_ids(db, 'gpp_genes')
	dip_genes = get_gene_ids(db, 'dip_genes')

	print('DOMINO Genes (i.e. "main" gene dataset): {:,}'.format(len(domino_genes)))
	main_gg = GeneGroups(db, valid_gene_ids=domino_genes)
	print('{} Train Disease: {:,}'.format(DR_T_NAME, len(main_gg.gpp_train_disease_gene_ids)))
	print('{} Train Non-Disease: {:,}'.format(DR_T_NAME, len(main_gg.gpp_train_tolerant_gene_ids)))

	print('{} Train AD: {:,}'.format(D_RT_NAME, len(main_gg.domino_train_ad_gene_ids)))
	print('{} Train AR: {:,}'.format(D_RT_NAME, len(main_gg.domino_train_ar_gene_ids)))	

	print('{} Validation AD: {:,}'.format(D_RT_NAME, len(main_gg.domino_validation_ad_gene_ids)))
	print('{} Validation AR: {:,}'.format(D_RT_NAME, len(main_gg.domino_validation_ar_gene_ids)))


	evaluation_genes = (set(domino_genes) - set(main_gg.gpp_train_disease_gene_ids) -
						set(main_gg.gpp_train_tolerant_gene_ids) - set(main_gg.domino_train_ad_gene_ids) -
						set(main_gg.domino_train_ar_gene_ids) - set(main_gg.domino_validation_ad_gene_ids) - 
						set(main_gg.domino_validation_ar_gene_ids)
					   )
	# The number of evaluation genes should be the same as length of DIP genes collection
	if len(evaluation_genes - set(dip_genes)) > 0:
		print('ERROR: number of evaluation genes calculated from scratch does not match number of DIP genes')
		return 0

	common_genes = set(evaluation_genes) & set(domino_genes) & set(gevir_genes) & set(gpp_genes)

	print('Evaluation genes: {:,}'.format(len(evaluation_genes)))
	print('Evaluation DOMINO & GPP & VIRLoF: {:,}'.format(len(common_genes)))

	eval_gg = GeneGroups(db, valid_gene_ids=evaluation_genes)
	common_gg = GeneGroups(db, valid_gene_ids=common_genes)

	print('Evaluation GDIT AD: {:,} ({:,})'.format(len(eval_gg.gdit_ad_gene_ids), len(common_gg.gdit_ad_gene_ids)))
	print('Evaluation GDIT AR: {:,} ({:,})'.format(len(eval_gg.gdit_ar_gene_ids), len(common_gg.gdit_ar_gene_ids)))
	print('Evaluation GDIT AD&AR: {:,} ({:,})'.format(len(eval_gg.gdit_ad_ar_gene_ids), len(common_gg.gdit_ad_ar_gene_ids)))
	print('Evaluation GDIT MD: {:,} ({:,})'.format(len(eval_gg.gdit_omim_gene_ids), len(common_gg.gdit_omim_gene_ids)))

	print('Evaluation GDIT Lethal AD: {:,} ({:,})'.format(len(eval_gg.gdit_lethal_ad_gene_ids), len(common_gg.gdit_lethal_ad_gene_ids)))
	print('Evaluation GDIT Lethal AR: {:,} ({:,})'.format(len(eval_gg.gdit_lethal_ar_gene_ids), len(common_gg.gdit_lethal_ar_gene_ids)))
	print('Evaluation GDIT Lethal AD&AR: {:,} ({:,})'.format(len(eval_gg.gdit_lethal_ad_ar_gene_ids), len(common_gg.gdit_lethal_ad_ar_gene_ids)))
	print('Evaluation GDIT Lethal MD: {:,} ({:,})'.format(len(eval_gg.gdit_lethal_omim_gene_ids), len(common_gg.gdit_lethal_omim_gene_ids)))

	print('Evaluation Cell Essential: {:,} ({:,})'.format(len(eval_gg.cell_essential_gene_ids), len(common_gg.cell_essential_gene_ids)))
	print('Evaluation Cell Non-Essential: {:,} ({:,})'.format(len(eval_gg.cell_non_essential_gene_ids), len(common_gg.cell_non_essential_gene_ids)))

	print('Evaluation Gene4Denovo: {:,} ({:,})'.format(len(eval_gg.gene4denovo_gene_ids),len(common_gg.gene4denovo_gene_ids)))
	print('Evaluation Severe HI: {:,} ({:,})'.format(len(eval_gg.severe_hi_gene_ids), len(common_gg.severe_hi_gene_ids)))
	print('Evaluation Olfactory: {:,} ({:,})'.format(len(eval_gg.olfactory_gene_ids), len(common_gg.olfactory_gene_ids)))

	print('Evaluation Mouse Het Lethal: {:,} ({:,})'.format(len(eval_gg.gdit_mouse_het_lethal_gene_ids), len(common_gg.gdit_mouse_het_lethal_gene_ids)))
	print('Evaluation Mouse Hom Lethal: {:,} ({:,})'.format(len(eval_gg.gdit_mouse_hom_lethal_gene_ids), len(common_gg.gdit_mouse_hom_lethal_gene_ids)))


##############################################
### RESULTS GENE RANKING COMPARISON TABLES ###
##############################################

def get_metrics_dict(db, valid_gene_ids=set(), use_dip_virlof=False):
	if use_dip_virlof:
		dip_collection_name = 'dip_virlof_genes'
	else:
		dip_collection_name = 'dip_genes'

	dip_scores = get_gene_scores(db, dip_collection_name, 'dip_rank', valid_gene_ids=valid_gene_ids, reverse=False)
	domino_scores = get_gene_scores(db, 'domino_genes', 'score', valid_gene_ids=valid_gene_ids, reverse=True)
	gpp_scores = get_gene_scores(db, 'gpp_genes', 'score', valid_gene_ids=valid_gene_ids, reverse=True)
	virlof_scores = get_gene_scores(db, 'gevir_genes', 'virlof', valid_gene_ids=valid_gene_ids, reverse=False)

	metrics_dict = OrderedDict()
	metrics_dict[DRT_NAME] = dip_scores.keys()
	metrics_dict['DOMINO'] = domino_scores.keys()
	metrics_dict['GPP'] = gpp_scores.keys()
	metrics_dict['VIRLoF'] = virlof_scores.keys()
	return metrics_dict


def get_metrics_subsets_dict(metrics_dict, n, top=True):
	metrics_subsets_dict = OrderedDict()
	for metric_name, gene_list in metrics_dict.items():
		gene_list = list(gene_list)
		if top:
			metric_subset = set(gene_list[:n])
		else:
			metric_subset = set(gene_list[len(gene_list) - n:])

		metrics_subsets_dict[metric_name] = metric_subset
	return metrics_subsets_dict


def get_single_gene_group_name_str(name, gene_set):
	return '{}, n = {}'.format(name, len(gene_set))
	

def get_double_gene_group_name_str(names, gene_sets):
	return '{}, n = {} ({}, n = {})'.format(names[0], len(gene_sets[0]), names[1], len(gene_sets[1]))


def get_gene_groups_dict(db, valid_gene_ids=set()):
	gg = GeneGroups(db, valid_gene_ids=valid_gene_ids)

	gene_groups_dict = OrderedDict()
	gene_groups_dict['AD'] = set(gg.gdit_ad_gene_ids)
	gene_groups_dict['AD (Lethal)'] = set(gg.gdit_lethal_ad_gene_ids)
	gene_groups_dict['AD&AR'] = set(gg.gdit_ad_ar_gene_ids)
	gene_groups_dict['AD&AR (Lethal)'] = set(gg.gdit_lethal_ad_ar_gene_ids)
	gene_groups_dict['AR'] = set(gg.gdit_ar_gene_ids)
	gene_groups_dict['AR (Lethal)'] = set(gg.gdit_lethal_ar_gene_ids)
	gene_groups_dict['MD'] = set(gg.gdit_omim_gene_ids)
	gene_groups_dict['MD (Lethal)'] = set(gg.gdit_lethal_omim_gene_ids)
	gene_groups_dict['ASD, EE or ID de novo'] = set(gg.gene4denovo_gene_ids)
	gene_groups_dict['Severe HI'] = set(gg.severe_hi_gene_ids)
	gene_groups_dict['Cell essential'] = set(gg.cell_essential_gene_ids)
	gene_groups_dict['Cell non-essential'] = set(gg.cell_non_essential_gene_ids)
	gene_groups_dict['Mouse Het Lethal'] = set(gg.gdit_mouse_het_lethal_gene_ids)
	gene_groups_dict['Mouse Hom Lethal'] = set(gg.gdit_mouse_hom_lethal_gene_ids)
	gene_groups_dict['Olfactory'] = set(gg.olfactory_gene_ids)

	return gene_groups_dict


def create_gene_metrics_evaluation_table(db, report_type):
	common_gene_ids = get_evaluation_common_gene_ids(db)
	metrics_dict = get_metrics_dict(db, valid_gene_ids=common_gene_ids)
	gene_groups_dict = get_gene_groups_dict(db, valid_gene_ids=common_gene_ids)
	
	if report_type == 'top25':
		n = round(len(common_gene_ids) * 0.25)
		metrics_subsets_dict = get_metrics_subsets_dict(metrics_dict, n, top=True)
	elif report_type == 'top10':
		n = round(len(common_gene_ids) * 0.1)
		metrics_subsets_dict = get_metrics_subsets_dict(metrics_dict, n, top=True)
	elif report_type == 'last25':
		n = round(len(common_gene_ids) * 0.25)
		metrics_subsets_dict = get_metrics_subsets_dict(metrics_dict, n, top=False)

	title = '{}, n={}/{}'.format(report_type, n, len(common_gene_ids))

	metric_names = list(metrics_subsets_dict.keys())
	headers_1 = ['Gene Group\\Metric']
	headers_2 = ['']
	for metric_name in metric_names:
		headers_1 += [metric_name, '']
		headers_2 += ['n', '%']
		if metric_name != DRT_NAME:
			headers_1.append('')
			headers_2.append('P')
	
	table = [[title], headers_1, headers_2]
	for group_name, gene_set in gene_groups_dict.items():
		row = [group_name]
		group_gene_num = len(gene_set)
		group_and_gr_metric_gene_num = len(gene_set & metrics_subsets_dict[DRT_NAME])	

		for metric_name, metric_subset_genes in metrics_subsets_dict.items():
			group_and_metric_gene_num = len(gene_set & metric_subset_genes)	
			group_ratio = get_str_ratio(group_and_metric_gene_num, group_gene_num, only_ratio=True)
			row.append(group_and_metric_gene_num) # n
			row.append(group_ratio) # %

			if metric_name != DRT_NAME:
				fe, p = fisher_exact([[group_and_gr_metric_gene_num, group_gene_num],
									  [group_and_metric_gene_num, group_gene_num]])
				p = '{:.2E}'.format(p) # P
				row.append(p)

		table.append(row)

	if report_type == 'last25':
		ar_set = gene_groups_dict['AR']
		all_ar_gene_num = len(ar_set)
		all_gene_num = len(common_gene_ids)

		domino_set = metrics_subsets_dict['DOMINO']
		domino_gene_num = len(domino_set)
		domino_ar_num = len(domino_set & ar_set)

		
		fe, p = fisher_exact([[domino_ar_num, domino_gene_num],
							  [all_ar_gene_num, all_gene_num]])

		print('Last 25% DOMINO AR Enrichment report:')
		print('AR/ALL = {}/{}'.format(all_ar_gene_num, all_gene_num))
		print('DOMINO AR/DOMINO Last 25% = {}/{}'.format(domino_ar_num, domino_gene_num))
		print('Fold Enrichment = {}; P = {:.2E}'.format(fe, p))
	
	report_name = report_type + '.csv'
	report_csv = TABLE_FOLDER / report_name
	write_table_to_csv(table, report_csv)

	# Calculate Precision Recall and F1 Statistics
	if report_type == 'top25' or report_type == 'top10':
		ad_set = gene_groups_dict['AD']
		ad_ar_set = gene_groups_dict['AD&AR']
		ar_set = gene_groups_dict['AR']
		ad_all_num = len(ad_set)
		ad_ar_all_num = len(ad_ar_set)
		stat_names = ['Precision', 'Recall', 'F1']
		headers_1 = ['Metirc', 'AR&AR ignored', '', '', 'AD&AR = FP', '', '', 'AD&AR = TP', '', '']
		headers_2 = [''] + stat_names * 3
		table = [headers_1, headers_2]
		for metric_name, metric_subset_genes in metrics_subsets_dict.items():
			row = [metric_name]
			ad_num = len(ad_set & metric_subset_genes)
			ad_ar_num = len(ad_ar_set & metric_subset_genes)
			ar_num = len(ar_set & metric_subset_genes)
			
			# AD&AR ignored
			stats = calculate_clf_performance(ad_num, ar_num, ad_all_num)
			for stat_name in stat_names:
				row.append(stats[stat_name])
			# AD&AR = FP
			stats = calculate_clf_performance(ad_num, ar_num+ad_ar_num, ad_all_num)
			for stat_name in stat_names:
				row.append(stats[stat_name])
			# AD&AR = TP
			stats = calculate_clf_performance(ad_num+ad_ar_num, ar_num, ad_all_num+ad_ar_all_num)
			for stat_name in stat_names:
				row.append(stats[stat_name])
			table.append(row)

		report_name = report_type + '_ml_stats.csv'
		report_csv = TABLE_FOLDER / report_name
		write_table_to_csv(table, report_csv)


def report_fisher_exact_statistics(first_name, first_pair, second_name, second_pair):
	fe, p = fisher_exact([first_pair, second_pair])
	print('{}: {}/{}'.format(first_name, first_pair[0], first_pair[1]))
	print('{}: {}/{}'.format(second_name, second_pair[0], second_pair[1]))
	print('Fold Enrichment = {}; P = {:.2E}'.format(fe, p))	


def report_metric_predicted_gene_nums(db):
	print('##################################')
	print('### Metrics Predicted Gene Num ###')
	print('##################################')	
	common_gene_ids = get_evaluation_common_gene_ids(db)
	#metrics_dict = get_metrics_dict(db, valid_gene_ids=common_gene_ids)

	domino_scores = get_gene_scores(db, 'domino_genes', 'score', valid_gene_ids=common_gene_ids, reverse=True)
	gpp_scores = get_gene_scores(db, 'gpp_genes', 'score', valid_gene_ids=common_gene_ids, reverse=True)

	domino_pred_ad_num = len(get_keys_from_dict_based_on_value_threshold(domino_scores, 0.5, '>='))
	gpp_pred_non_disease_num = len(get_keys_from_dict_based_on_value_threshold(gpp_scores, 0.5, '<='))

	dip_scores = get_gene_scores(db, 'dip_genes', 'dip_rank', valid_gene_ids=common_gene_ids, reverse=False)
	common_dip_gene_ids = list(dip_scores.keys())
	all_dip_gene_ids = get_gene_ids(db, 'dip_genes')
	
	all_dip_non_disease_gene_ids = set()
	dip_non_disease_genes = db.main.dip_genes.find({ 't_prob': { '$gte': 0.5 } })
	for dip_non_disease_gene in dip_non_disease_genes:
		all_dip_non_disease_gene_ids.add(dip_non_disease_gene['hgnc_gene_id'])

	common_dip_non_disease_gene_ids = all_dip_non_disease_gene_ids & set(common_dip_gene_ids)

	#dip_pred_non_disease_num = dip_non_disease_genes.count()

	# Genes sorted by DND model (cannot be predicted by ADR model to be disease in the final ranking)
	#dr_t_ranked_genes = set(all_dip_gene_ids[round(len(all_dip_gene_ids) / 2):])
	d_rt_genes = db.main.dip_genes.find({ 'd_prob': { '$gte': 0.5 } })

	d_rt_pred_ad_gene_ids = set()
	first_half_d_rt_pred_ad_gene_ids = set()
	for d_rt_gene in d_rt_genes:
		gene_id = d_rt_gene['hgnc_gene_id']
		d_rt_pred_ad_gene_ids.add(gene_id)

		if d_rt_gene['dip_rank'] < 50:
			first_half_d_rt_pred_ad_gene_ids.add(gene_id)
	
	common_dip_ad_gene_ids = first_half_d_rt_pred_ad_gene_ids & set(common_gene_ids)

	common_gene_num = len(common_gene_ids)
	# Output stats for common evaluation dataset
	print('Common evaluation dataset gene number:', common_gene_num)
	print('GPP Non-disease: ' + get_str_ratio(gpp_pred_non_disease_num, common_gene_num))
	print('DND Non-disease: ' + get_str_ratio(len(common_dip_non_disease_gene_ids), common_gene_num))
	print('DOMINO AD:' + get_str_ratio(domino_pred_ad_num, common_gene_num))
	print('ADR (-DND) AD:' + get_str_ratio(len(common_dip_ad_gene_ids), common_gene_num))
	print('##################################')
	dip_gene_num = len(all_dip_gene_ids)
	print('Evaluation dataset gene number:', dip_gene_num)
	print('ADR (-DND) AD:' + get_str_ratio(len(first_half_d_rt_pred_ad_gene_ids), common_gene_num))
	print('DND Non-disease (>=0.5): ' + get_str_ratio(len(all_dip_non_disease_gene_ids), dip_gene_num))
	print('ADR AD (>=0.5): ' + get_str_ratio(len(d_rt_pred_ad_gene_ids), dip_gene_num))
	adr_and_dnd_gene_ids = d_rt_pred_ad_gene_ids & all_dip_non_disease_gene_ids
	print('ADR & DND (>=0.5) out of all ADR pred genes: ' + get_str_ratio(len(adr_and_dnd_gene_ids), len(d_rt_pred_ad_gene_ids)))
	'''
	d_rt_pred_ad_gene_num = len(d_rt_pred_ad_gene_ids)
	dip_pred_ad_gene_ids = d_rt_pred_ad_gene_ids - dr_t_ranked_genes
	dip_pred_ad_gene_num = len(dip_pred_ad_gene_ids)
	excluded_dip_pred_ad_gene_ids = d_rt_pred_ad_gene_ids & dr_t_ranked_genes
	ex_gene_num = len(excluded_dip_pred_ad_gene_ids)
	total_gene_num = len(common_gene_ids)
	dip_gene_num = len(dip_gene_ids)
	'''

	#print('##################################')
	#print('### Metrics Predicted Gene Num ###')
	#print('##################################')
	#print('Evaluation dataset gene number (common):', total_gene_num)
	#print('GPP Non-disease: ' + get_str_ratio(gpp_pred_non_disease_num, total_gene_num))
	
	'''
	print('Evaluation dataset gene number (non-common):', dip_gene_num)
	print('DND Non-disease: ' + get_str_ratio(len(dip_non_disease_gene_ids), dip_gene_num))
	print('ADR AD: ' + get_str_ratio(d_rt_pred_ad_gene_num, dip_gene_num))
	print('ADR - DND AD: ' + get_str_ratio(dip_pred_ad_gene_num, dip_gene_num))
	'''
	print('################################################')
	print('### ADR & DND (prob >= 0.5) enrichment stats ###')
	print('################################################')
	gg = GeneGroups(db, valid_gene_ids=all_dip_gene_ids)
	#print('ADR - DND:', ex_gene_num)
	gene_groups = OrderedDict()
	gene_groups['AD'] = set(gg.gdit_ad_gene_ids)
	gene_groups['AD&AR'] = set(gg.gdit_ad_ar_gene_ids)
	gene_groups['AR'] = set(gg.gdit_ar_gene_ids)
	gene_groups['Cell non-essential'] = set(gg.cell_non_essential_gene_ids)
	gene_groups['Olfactory'] = set(gg.olfactory_gene_ids)

	for gene_group_name, gene_group_gene_ids in gene_groups.items():
		adr_and_dnd_and_group_gene_num = len(adr_and_dnd_gene_ids & gene_group_gene_ids)
		first_pair = [adr_and_dnd_and_group_gene_num, len(adr_and_dnd_gene_ids)]
		group_gene_num = len(gene_group_gene_ids)
		second_pair = [group_gene_num, dip_gene_num]
		print(gene_group_name)
		report_fisher_exact_statistics('ADR>=0.5 & DND > 50%', first_pair, 
									   'ALL', second_pair)


def report_gene_median_stats(metric_name, gene_ids, gene_id_to_prop_dict):
	values = []
	for gene_id in gene_ids:
		if gene_id in gene_id_to_prop_dict:
			values.append(gene_id_to_prop_dict[gene_id])
	not_found_genes = len(gene_ids) - len(values)
	median = np.median(values)
	print(metric_name, 'Median:', median)
	print(metric_name, 'Genes NOT FOUND:', not_found_genes)


def report_dip_vs_domino_top_25_overlap(db):
	# combined_score_500
	# textmining_500
	print('REPORT DIP vs DOMINO')

	common_gene_ids = get_evaluation_common_gene_ids(db)
	metrics_dict = get_metrics_dict(db, valid_gene_ids=common_gene_ids)
	n = round(len(common_gene_ids) * 0.25)
	metrics_subsets_dict = get_metrics_subsets_dict(metrics_dict, n, top=True)

	dip_genes = set(list(metrics_subsets_dict[DRT_NAME]))
	domino_genes = set(list(metrics_subsets_dict['DOMINO']))

	total_genes = len(dip_genes)
	gene_id_to_virlof = get_gene_id_to_virlof(db)
	gene_id_to_string_ppi_num = get_gene_id_to_string_ppi_num(db, 'combined_score_500')

	genes_in_common = dip_genes & domino_genes
	dip_not_domino = dip_genes - domino_genes
	domino_not_dip = domino_genes - dip_genes
	print('Common genes: {} ({:.2f}%)'.format(len(genes_in_common), len(genes_in_common) * 100 / total_genes))
	print('Diff genes: {} ({:.2f}%)'.format(len(dip_not_domino), len(dip_not_domino) * 100 / total_genes))
	print('DIP not DOMINO')
	report_gene_median_stats('VIRLoF', dip_not_domino, gene_id_to_virlof)
	report_gene_median_stats('STRING combined_score_500', dip_not_domino, gene_id_to_string_ppi_num)
	print('DOMINO not DIP')
	report_gene_median_stats('VIRLoF', domino_not_dip, gene_id_to_virlof)
	report_gene_median_stats('STRING combined_score_500', domino_not_dip, gene_id_to_string_ppi_num)

	all_dip_genes = get_metric_ranked_gene_scores(db, 'dip_genes', 'dip_rank', reverse=False)
	all_domino_genes = get_metric_ranked_gene_scores(db, 'domino_genes', 'score', reverse=True)

	# Export DIP vs DOMINO genes
	rows = []
	gg = GeneGroups(db)
	for gene_id in (dip_genes | domino_genes):
		gene_data = OrderedDict()
		gene_data['hgnc_gene_id'] = gene_id
		gene_data['hgnc_gene_name'] = gg.gene_id_to_gene_name[gene_id]
		gene_data['dip_rank'] = all_dip_genes[gene_id]
		gene_data['domino_score'] = all_domino_genes[gene_id]
		gene_data['virlof'] = gene_id_to_virlof[gene_id]
		gene_data['string_ppi_num'] = gene_id_to_string_ppi_num[gene_id]
		dip_vs_domino_group = ''
		if gene_id in genes_in_common:
			dip_vs_domino_group = 'both'
		if gene_id in dip_not_domino:
			dip_vs_domino_group = 'dip'
		if gene_id in domino_not_dip:
			dip_vs_domino_group = 'domino'
		gene_data['dip_vs_domino_group'] = dip_vs_domino_group

		headers = list(gene_data.keys())
		rows.append(list(gene_data.values()))

	table = [headers] + rows
	report_csv = TABLE_FOLDER / 'dip_vs_domino_top_genes.csv'
	write_table_to_csv(table, report_csv)


###############################
### FOLD ENRCHMENT ANALYSIS ###
###############################

def create_all_metrics_fold_enrichment_dataset(db):
	common_gene_ids = get_evaluation_common_gene_ids(db)
	metrics_dict = get_metrics_dict(db, valid_gene_ids=common_gene_ids, use_dip_virlof=True)

	gg = GeneGroups(db, valid_gene_ids=common_gene_ids)

	ad_set = set(gg.gdit_ad_gene_ids)
	ar_set = set(gg.gdit_ar_gene_ids)
	ad_ar_set = set(gg.gdit_ad_ar_gene_ids)
	cell_non_essential_set = set(gg.cell_non_essential_gene_ids)

	fe_genes = {}
	for gene_id in common_gene_ids:
		fe_gene = OrderedDict()
		fe_gene['hgnc_gene_id'] = gene_id
		fe_genes[gene_id] = fe_gene
		for metric_name in metrics_dict.keys():
			fe_genes[gene_id][metric_name] = OrderedDict()


	for metric_name, metric_gene_ids in metrics_dict.items():
		metric_gene_ids = list(metric_gene_ids)
		print(metric_name)
		print('Calculating AD FE stats...')
		ad_stats = get_ranked_gene_enrichment_stats(metric_gene_ids, ad_set, 5)
		print('Calculating AD+AR FE stats...')
		ad_ar_stats = get_ranked_gene_enrichment_stats(metric_gene_ids, ad_ar_set, 5)
		print('Calculating AR FE stats...')
		ar_stats = get_ranked_gene_enrichment_stats(metric_gene_ids, ar_set, 5)
		print('Cell-non-essential FE stats...')
		cell_non_essential_stats = get_ranked_gene_enrichment_stats(metric_gene_ids, cell_non_essential_set, 5)

		for gene_id in common_gene_ids:
			fe_stats = OrderedDict()
			fe_stats['fe_examined_range'] = ad_stats[gene_id]['examined_range']
			fe_stats['ad_fe'] = ad_stats[gene_id]['fold_enrichment']
			fe_stats['ad_p'] = ad_stats[gene_id]['p_value']
			fe_stats['ad_ar_fe'] = ad_ar_stats[gene_id]['fold_enrichment']
			fe_stats['ad_ar_p'] = ad_ar_stats[gene_id]['p_value']
			fe_stats['ar_fe'] = ar_stats[gene_id]['fold_enrichment']
			fe_stats['ar_p'] = ar_stats[gene_id]['p_value']
			fe_stats['cell_non_essential_fe'] = cell_non_essential_stats[gene_id]['fold_enrichment']
			fe_stats['cell_non_essential_p'] = cell_non_essential_stats[gene_id]['p_value']
			fe_genes[gene_id][metric_name] = fe_stats

	db.main.common_eval_genes_fe.drop()
	db.main.common_eval_genes_fe.insert_many(list(fe_genes.values()))
	db.main.common_eval_genes_fe.create_index([('hgnc_gene_id', pymongo.ASCENDING)], name='hgnc_gene_id_1')


def report_fold_enrichment_peak(db, score_name, group_name):
	field_name = score_name + '.' + group_name + '_fe'
	peak_gene = db.main.common_eval_genes_fe.find_one(sort=[(field_name, -1)])

	print(score_name, group_name)
	print('Range:', peak_gene[score_name]['fe_examined_range'])
	print('FE:{:.2f}'.format(peak_gene[score_name][group_name + '_fe']))
	print('P:{:.2E}'.format(peak_gene[score_name][group_name + '_p']))
	print('#######################')


def report_fold_enrichment_peaks(db):
	report_fold_enrichment_peak(db, 'DIP', 'ad')
	report_fold_enrichment_peak(db, 'DIP', 'ad_ar')
	report_fold_enrichment_peak(db, 'DIP', 'ar')
	report_fold_enrichment_peak(db, 'DIP', 'cell_non_essential')

	report_fold_enrichment_peak(db, 'VIRLoF', 'ar')
	report_fold_enrichment_peak(db, 'DOMINO', 'ad')
	report_fold_enrichment_peak(db, 'GPP', 'cell_non_essential')


###############################
### PAPER STATISTICAL TESTS ###
###############################

def report_ad_proportion_in_domino_datasets(db):
	gg = GeneGroups(db)
	train_ad_prop = '{:.2f}%'.format(len(gg.domino_train_ad_gene_ids) * 100 / len(gg.domino_train_gene_ids))
	validation_ad_prop = '{:.2f}%'.format(len(gg.domino_validation_ad_gene_ids) * 100 / len(gg.domino_validation_gene_ids))
	print('DOMINO Train AD | All | %')
	print(len(gg.domino_train_ad_gene_ids), len(gg.domino_train_gene_ids), train_ad_prop)
	print('DOMINO Validaton AD | All | %')
	print(len(gg.domino_validation_ad_gene_ids), len(gg.domino_validation_gene_ids), validation_ad_prop)


def compare_d_rt_train_and_evaluaton_or_validation_ad_gene_virlof(db, alt_ad_name):
	gene_id_to_virlof = get_gene_id_to_virlof(db)
	
	gg = GeneGroups(db)
	ad_train = set(gg.domino_train_ad_gene_ids)
	
	if alt_ad_name == 'evaluation':
		dip_gene_ids = set(get_gene_ids(db, 'dip_genes'))
		gg = GeneGroups(db, valid_gene_ids=dip_gene_ids)
		ad_alt = gg.gdit_ad_gene_ids
	elif alt_ad_name == 'validation':
		ad_alt = gg.domino_validation_ad_gene_ids

	print('AD genes: train | ' + alt_ad_name)
	print(len(ad_train), len(ad_alt))

	ad_train_virlof = []
	for gene_id in ad_train:
		if gene_id in gene_id_to_virlof:
			ad_train_virlof.append(gene_id_to_virlof[gene_id])
	
	ad_evaluation_virlof = []
	for gene_id in ad_alt:
		if gene_id in gene_id_to_virlof:
			ad_evaluation_virlof.append(gene_id_to_virlof[gene_id])

	print('AD genes with VIRLoF: train | ' + alt_ad_name)
	print(len(ad_train_virlof), len(ad_evaluation_virlof))
	print('AD genes median VIRLoF: train | ' + alt_ad_name)
	print(np.median(ad_train_virlof), np.median(ad_evaluation_virlof))
	statistic, p_value = mannwhitneyu(ad_train_virlof, ad_evaluation_virlof, alternative='two-sided')
	print('Mann Whitney U test two-sided with continuity correction (1/2.): statistic | p-value')
	print(statistic, p_value)


def compare_ad_genes_predicted_by_ppi_models_and_final_dip(db):
	dip_gene_ids = set(get_gene_ids(db, 'dip_genes'))
	gg = GeneGroups(db, valid_gene_ids=dip_gene_ids)

	f = Features(db)
	ppi_n1_probs = f.get_gene_id_to_value_dict(dip_gene_ids, 'string_first_neighbour_ad_knn_prob')
	ppi_n2_probs = f.get_gene_id_to_value_dict(dip_gene_ids, 'string_second_neighbour_ad_knn_prob')

	ppi_n1_pred_gene_ids = set()
	ppi_n2_pred_gene_ids = set()

	ppi_n1_high_prob_gene_ids = set()
	ppi_n2_high_prob_gene_ids = set()
	for gene_id in dip_gene_ids:
		if ppi_n1_probs[gene_id] >= 0.5:
			ppi_n1_pred_gene_ids.add(gene_id)
		if ppi_n2_probs[gene_id] >= 0.5:
			ppi_n2_pred_gene_ids.add(gene_id)
		if ppi_n1_probs[gene_id] >= 0.9:
			ppi_n1_high_prob_gene_ids.add(gene_id)
		if ppi_n2_probs[gene_id] >= 0.9:
			ppi_n2_high_prob_gene_ids.add(gene_id)

	print('N1 pred AD:', len(ppi_n1_pred_gene_ids))
	print('N2 pred AD:', len(ppi_n2_pred_gene_ids))

	dip_genes = db.main.dip_genes.find({ "d_prob": { "$gte": 0.5 }, "dip_rank": { "$lt": 50.0 } })
	dip_pred_ad_gene_ids = set()

	for dip_gene in dip_genes:
		dip_pred_ad_gene_ids.add(dip_gene['hgnc_gene_id'])

	
	print('DIP pred AD:', len(dip_pred_ad_gene_ids))
	print('DIP & N1 pred AD:', len(dip_pred_ad_gene_ids & ppi_n1_pred_gene_ids))
	print('DIP & N2 pred AD:', len(dip_pred_ad_gene_ids & ppi_n2_pred_gene_ids))

	print('Proportion of N1, N2 predicted genes also predicted by DIP')
	print('N1', get_str_ratio(len(dip_pred_ad_gene_ids & ppi_n1_pred_gene_ids), len(ppi_n1_pred_gene_ids)))
	print('N2', get_str_ratio(len(dip_pred_ad_gene_ids & ppi_n2_pred_gene_ids), len(ppi_n2_pred_gene_ids)))

	n1_gene_ppi_num = {}
	n2_gene_ppi_num = {}
	string_genes = db.main.string_genes_v10.find({})
	for string_gene in string_genes:
		gene_id = string_gene['hgnc_gene_id']
		if gene_id in dip_gene_ids:
			n1_gene_ppi_num[gene_id] = len(string_gene['textmining_500'])
			n2_gene_ppi_num[gene_id] = len(string_gene['textmining_500'])

	median_gene_ppi = np.median(list(n1_gene_ppi_num.values()))
	print('Median PPI Num:', median_gene_ppi)

	low_ppi_n1_pred_gene_ids = set()
	for gene_id in ppi_n1_pred_gene_ids:
		if n1_gene_ppi_num[gene_id] <= median_gene_ppi:
			low_ppi_n1_pred_gene_ids.add(gene_id)

	low_ppi_n2_pred_gene_ids = set()
	for gene_id in ppi_n2_pred_gene_ids:
		if n2_gene_ppi_num[gene_id] <= median_gene_ppi:
			low_ppi_n2_pred_gene_ids.add(gene_id)

	print('Proportion of low PPI from all gene predicted by N1, N2')
	print('N1', get_str_ratio(len(low_ppi_n1_pred_gene_ids), len(ppi_n1_pred_gene_ids)))
	print('N2', get_str_ratio(len(low_ppi_n2_pred_gene_ids), len(ppi_n2_pred_gene_ids)))

	print('Proportion of low PPI N1, N2 predicted genes also predicted by DIP')
	print('N1', get_str_ratio(len(dip_pred_ad_gene_ids & low_ppi_n1_pred_gene_ids), len(low_ppi_n1_pred_gene_ids)))
	print('N2', get_str_ratio(len(dip_pred_ad_gene_ids & low_ppi_n2_pred_gene_ids), len(low_ppi_n2_pred_gene_ids)))


def report_severe_hi_virlof(db):
	common_gene_ids = get_evaluation_common_gene_ids(db)
	gene_id_to_virlof = get_gene_id_to_virlof(db)
	gg = GeneGroups(db, valid_gene_ids=common_gene_ids)

	severe_hi_virlof = []
	for gene_id in gg.severe_hi_gene_ids:
		if gene_id in gene_id_to_virlof:
			severe_hi_virlof.append(gene_id_to_virlof[gene_id])
		else:
			print(gene_id)
	print('Evaluation dataset Severe HI median VIRLoF:', np.median(severe_hi_virlof))


def main():
	# Uncomment lines to run the analysis
	db = MongoDB()

	#report_methods_genes(db)
	
	# Reports number of genes predicted by each method (GPP, DOMINO, ADR, DND)
	#report_metric_predicted_gene_nums(db)
	#report_dip_vs_domino_top_25_overlap(db)

	#create_all_metrics_fold_enrichment_dataset(db)
	#report_ad_proportion_in_domino_datasets(db)
	#compare_d_rt_train_and_evaluaton_or_validation_ad_gene_virlof(db, 'validation')
	#compare_d_rt_train_and_evaluaton_or_validation_ad_gene_virlof(db, 'evaluation')
	#compare_ad_genes_predicted_by_ppi_models_and_final_dip(db)

	# Methods for STRING PPI analysis are in string_ppi.py
	#analyse_domino_direct_ad_method_performance(db)
	#analyse_string_knn_method_performance(db)

	# Methods to produce tables with ADR model performance on
	# DOMINO Train (10x10 cross-validation) and Validation datasets
	#train_cv_d_rt_model(db, report_domino=True)
	#test_d_rt_model(db, 'domino_validation', report_domino=True)

	#report_severe_hi_virlof(db)

	# Metric Fold Enrichment Peaks
	#report_fold_enrichment_peaks(db)


	# Paper Tables 1, 2, and 3
	#create_gene_metrics_evaluation_table(db, 'top25')
	#create_gene_metrics_evaluation_table(db, 'top10')
	#create_gene_metrics_evaluation_table(db, 'last25')

	# Paper Supplementary Table 1
	#export_gene_metrics(db, include_omim_original=False, use_dip_virlof=True)


if __name__ == "__main__":
	sys.exit(main())