import os
import sys
import progressbar
import pymongo
import numpy as np

from pathlib import Path
from scipy.stats import fisher_exact

from features import Features
from clf_perf import ClfPerf, ClfResult, get_class_probs
from clf_perf import report_clfs_mean_metrics_comparison
from progressbar import ProgressBar, ETA, Percentage, Bar
from collections import OrderedDict
from common import MongoDB, sort_dict_by_values, calculate_percentiles, get_str_ratio
from common import write_table_to_csv, report_gene_group_enrichment_in_the_subset
from common import get_metric_ranked_gene_scores
from common import calculate_clf_performance
from gene_groups import GeneGroups, get_common_domino_gpp_and_gevir_gene_ids
from gene_groups import get_gene_id_to_virlof, get_gene_id_to_string_ppi_num
from feature_selection import get_model_clf_and_feature_names, get_all_feature_names

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


import matplotlib.pyplot as plt

PROGRESS_BAR_WIDGETS = [Percentage(), ' ', Bar(), ' ', ETA()]

TABLE_FOLDER = Path('tables/')

DR_T_NAME = 'DR_T'
D_RT_NAME = 'D_RT'
 
#################
### CONSTANTS ###
#################

CV_N_FOLDS = 10
CV_N_RUNS = 10

##############
### DOMINO ###
##############

class Domino():
	def __init__(self, db):
		self.gg = GeneGroups(db)
		self.gene_ad_probs = {}
		self.gene_ar_probs = {}
		self.gene_preds = {}

		domino_genes = db.main.domino_genes.find({})
		for domino_gene in domino_genes:
			gene_id = domino_gene['hgnc_gene_id']
			ad_prob = domino_gene['score']
			ar_prob = 1 - ad_prob

			if ad_prob >= 0.5:
				pred = 0
			else:
				pred = 1

			self.gene_ad_probs[gene_id] = ad_prob
			self.gene_ar_probs[gene_id] = ar_prob
			self.gene_preds[gene_id] = pred


	def _get_performance(self, gene_ids, labels):
		clf_perf = ClfPerf(clf_name='DOMINO')
		pred_y = []
		prob_y = []
		for gene_id in gene_ids:
			pred_y.append(self.gene_preds[gene_id])
			prob_y.append(self.gene_ar_probs[gene_id])

		clf_perf.update_perf_metrics(labels, pred_y, y_prob=prob_y, sample_ids=gene_ids)
		clf_perf.calculate_mean_metrics()
		return clf_perf


	def get_clf_perf_domino_train(self, report=False):
		clf_perf = self._get_performance(self.gg.domino_train_gene_ids, self.gg.domino_train_labels)
		if report:
			clf_perf.report_performance()
		return clf_perf


	def get_clf_perf_domino_validation(self, report=False):
		clf_perf = self._get_performance(self.gg.domino_validation_gene_ids, 
										 self.gg.domino_validation_labels)
		if report:
			clf_perf.report_performance()
		return clf_perf


	def get_clf_perf_gdit_test(self, report=False):
		clf_perf = self._get_performance(self.gg.gdit_test_gene_ids, self.gg.gdit_test_labels)
		if report:
			clf_perf.report_performance()
		return clf_perf

##########
### MY ###
##########

def get_model_data(db, model_name):
	gg = GeneGroups(db)
	if model_name == 'd_rt':
		clf_perf = ClfPerf(clf_name='D_RT', class_names=('AD', 'AR'))
		gene_ids = np.array(gg.domino_train_gene_ids)
		labels = np.array(gg.domino_train_labels)
	elif model_name == 'dr_t':
		clf_perf = ClfPerf(clf_name='DR_T', class_names=('Null', 'AR'))
		gene_ids = np.array(gg.gpp_train_gene_ids)
		labels = np.array(gg.gpp_train_labels)
	return clf_perf, gene_ids, labels


#####################################
### TRAINING and CROSS-VALIDATION ###
#####################################

def train_cv_dr_t_model(db):
	print('DR_T Model Cross-Validation')
	clf, feature_names = get_model_clf_and_feature_names(db, 'dr_t')
	clf_perf = run_cv_model(db, clf, feature_names, model_name='dr_t')
	clf_perf.report_performance()


def train_cv_d_rt_model(db, report_domino=False):
	print('D_RT Model Cross-Validation')
	if report_domino:
		domino = Domino(db)
		domino_clf_perf = domino.get_clf_perf_domino_train(report=False)

	# Note: textmining and experimental features together work better on training and GDIT testing, 
	# but not validation
	clf, feature_names = get_model_clf_and_feature_names(db, 'd_rt')
	clf_perf = run_cv_model(db, clf, feature_names)

	if report_domino:
		report_csv = TABLE_FOLDER / 'dr_t_domino_train_cv_performance.csv'
		report_clfs_mean_metrics_comparison(clf_perf, domino_clf_perf, report_csv=report_csv)
	else:
		clf_perf.report_performance()


def run_cv_model(db, clf, feature_names, model_name='d_rt'):
	clf_perf, gene_ids, labels = get_model_data(db, model_name)
	f = Features(db)
	features = np.array(f.get_values(gene_ids, feature_names))

	# KF with 0 random seed is used to find best features/clf paramters and might be overfitted.
	random_seeds = range(1, CV_N_RUNS + 1) 
	n_folds = CV_N_FOLDS
	max_exp = CV_N_RUNS * CV_N_FOLDS
	exp_num = 0
	bar = progressbar.ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()
	for random_seed in random_seeds:
		kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
		for train, test in kf.split(gene_ids, labels):
			test_gene_ids = gene_ids[test]

			X_train = features[train]
			X_test = features[test]

			y_train = labels[train]
			y_test = labels[test]

			clf.fit(X_train, y_train)
			pred_y = clf.predict(X_test)
			prob_y = clf.predict_proba(X_test)
			prob_main_class = get_class_probs(prob_y, 0)
			prob_second_class = get_class_probs(prob_y, 1)
			
			clf_perf.update_perf_metrics(y_test, pred_y, y_prob=prob_second_class, 
				prob_main_class=prob_main_class, sample_ids=test_gene_ids)
			exp_num += 1

			bar.update((exp_num + 0.0) / max_exp)
	bar.finish()

	clf_perf.calculate_mean_metrics()
	return clf_perf

###############
### TESTING ###
###############

def test_d_rt_model(db, dataset_name, report_domino=False):
	print('D_RT Model Testing on', dataset_name)
	if report_domino:
		domino = Domino(db)
		if dataset_name == 'domino_validation':
			domino_clf_perf = domino.get_clf_perf_domino_validation(report=False)
		elif dataset_name == 'gdit_test':
			domino_clf_perf = domino.get_clf_perf_gdit_test(report=False)

	gg = GeneGroups(db)
	f = Features(db)

	clf_perf = ClfPerf(clf_name='D_RT')

	clf, feature_names = get_model_clf_and_feature_names(db, 'd_rt')

	y_train = gg.domino_train_labels
	X_train = f.get_values(gg.domino_train_gene_ids, feature_names)

	if dataset_name == 'domino_validation':
		y_test = gg.domino_validation_labels
		X_test = f.get_values(gg.domino_validation_gene_ids, feature_names)
		test_gene_ids = gg.domino_validation_gene_ids
	elif dataset_name == 'gdit_test':
		y_test = gg.gdit_test_labels
		X_test = f.get_values(gg.gdit_test_gene_ids, feature_names)
		test_gene_ids = gg.gdit_test_gene_ids

	clf.fit(X_train, y_train)

	pred_y = clf.predict(X_test)
	prob_y = clf.predict_proba(X_test)
	prob_y = get_class_probs(prob_y, 1)
	clf_perf.update_perf_metrics(y_test, pred_y, y_prob=prob_y, sample_ids=test_gene_ids)
	clf_perf.calculate_mean_metrics()

	if report_domino:
		report_csv = TABLE_FOLDER / ('dr_t_' + dataset_name + '_performance.csv')
		report_clfs_mean_metrics_comparison(clf_perf, domino_clf_perf, report_csv=report_csv)
	else:
		clf_perf.report_performance()

################################
### MODEL PERFORMANCE TABLES ###
################################

def get_gene_scores(db, collection_name, score_key, valid_gene_ids=set()):
	gene_scores = {}
	genes = db.main[collection_name].find({})
	for gene in genes:
		gene_id = gene['hgnc_gene_id']
		if gene_id in valid_gene_ids:
			gene_scores[gene_id] = gene[score_key]	
	return gene_scores

def get_top_gene_ids_from_gene_scores_dict(gene_scores, top_num, reverse=False):
	top_gene_ids = sort_dict_by_values(gene_scores, reverse=reverse).keys()[:top_num]


def export_dr_t_model_results(db, include_uneecon=False):
	valid_gene_ids = get_common_domino_gpp_and_gevir_gene_ids(db, no_gpp_train=True, include_uneecon=include_uneecon)

	gpp_t_gene_ids = set()
	gpp_genes = db.main.gpp_genes.find({ 'score': { '$lt': 0.5 } })
	for gpp_gene in gpp_genes:
		gpp_t_gene_ids.add(gpp_gene['hgnc_gene_id'])

	#print(len(gpp_t_gene_ids))
	gpp_t_gene_ids = gpp_t_gene_ids & valid_gene_ids
	#print(len(gpp_t_gene_ids))
	t_genes_num = len(gpp_t_gene_ids)

	domino_gene_scores = get_gene_scores(db, 'domino_genes', 'score', valid_gene_ids=valid_gene_ids)
	dr_t_gene_scores = get_gene_scores(db, 'drt_genes', 't_prob', valid_gene_ids=valid_gene_ids)
	virlof_gene_scores = get_gene_scores(db, 'gevir_genes', 'virlof', valid_gene_ids=valid_gene_ids)

	if include_uneecon:
		uneecon_gene_scores = get_gene_scores(db, 'uneecon_genes', 'score', valid_gene_ids=valid_gene_ids)


	domino_top_t_gene_ids = set(list(sort_dict_by_values(domino_gene_scores, reverse=False).keys())[:t_genes_num])
	dr_top_t_gene_ids = set(list(sort_dict_by_values(dr_t_gene_scores, reverse=True).keys())[:t_genes_num])
	virlof_top_t_gene_ids = set(list(sort_dict_by_values(virlof_gene_scores, reverse=True).keys())[:t_genes_num])
	if include_uneecon:
		uneecon_top_t_gene_ids = set(list(sort_dict_by_values(uneecon_gene_scores, reverse=False).keys())[:t_genes_num])

	metrics_top_t_sets = OrderedDict()
	metrics_top_t_sets[DR_T_NAME] = dr_top_t_gene_ids
	metrics_top_t_sets['GPP'] = gpp_t_gene_ids
	metrics_top_t_sets['VIRLoF'] = virlof_top_t_gene_ids
	metrics_top_t_sets['DOMINO'] = domino_top_t_gene_ids
	if include_uneecon:
		metrics_top_t_sets['UNEECON'] = uneecon_top_t_gene_ids

	metric_stats = OrderedDict()
	metric_fp_nums = OrderedDict()
	gg = GeneGroups(db, valid_gene_ids=valid_gene_ids)
	for metric_name, top_t_gene_ids in metrics_top_t_sets.items():
		fp_num = 0
		metric_gene_groups = OrderedDict()
		group_set = set(gg.gdit_ad_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		fp_num += group_and_top_t_gene_num # AD
		metric_gene_groups['AD ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.gdit_ad_ar_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		fp_num += group_and_top_t_gene_num # AD,AR
		metric_gene_groups['AD,AR ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.gdit_ar_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		fp_num += group_and_top_t_gene_num # AR
		metric_gene_groups['AR ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))		
		
		# Report AR Enrichment
		ar_num = len(group_set)
		ar_and_subset = group_and_top_t_gene_num
		subset_num = len(top_t_gene_ids)
		total_num = len(valid_gene_ids)
		ar_fe, ar_p = fisher_exact([[ar_and_subset, subset_num],
									[ar_num, total_num]])
		print('{} AR FE={}, P={}'.format(metric_name, ar_fe, ar_p)) 

		group_set = set(gg.cell_essential_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Cell Essential ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.cell_non_essential_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Cell Non-essential ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.olfactory_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Olfactory ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))

		# Extra sets for comparison
		group_set = set(gg.severe_hi_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Severe HI ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.gene4denovo_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Gene4Denovo ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.mouse_het_lethal_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Mouse Het Lethal ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))

		# ALL OMIM Variations
		group_set = set(gg.gdit_omim_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['GDIT OMIM ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))

		group_set = set(gg.gpp_pathogenic_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['GPP OMIM ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))

		group_set = set(gg.omim_original_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['MY OMIM ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))

		metric_stats[metric_name] = metric_gene_groups
		metric_fp_nums[metric_name] = fp_num

	title = 'Table 1. Analysis of ' + get_str_ratio(t_genes_num, len(valid_gene_ids)) + \
			' genes predicted to be non-disease by GPP (score < 0.5) and ' + \
			'the same number of genes with the highest predicted probability of being non-disease by ' + \
			DR_T_NAME + ' or the lowest VIRLoF and DOMINO scores' + \
			' out of ' + str(len(valid_gene_ids)) + ' genes which have predicted scores by all metrics ' + \
			'and were not used in GPP and ' + DR_T_NAME + ' model training.'

	metric_names = list(metric_stats.keys())
	gene_group_names = metric_stats[metric_names[0]].keys()
	headers = ['Gene Group\\Metric'] + metric_names
	table = [[title], headers]
	for gene_group_name in gene_group_names:
		row = [gene_group_name]
		for metric_name in metric_names:
			row.append(metric_stats[metric_name][gene_group_name])
		table.append(row)

	report_csv = TABLE_FOLDER / 'dr_t_performance.csv'
	write_table_to_csv(table, report_csv)

	# Compare DR_T performance with others on False Positives (FP)
	headers = ['Metric', '{} FP'.format(DR_T_NAME), 'Metric FP', 'Genes', 'Fold Enrichment', 'P value']
	table = [headers]
	dr_t_fp = metric_fp_nums[DR_T_NAME]
	for metric_name, fp_num in metric_fp_nums.items():
		if metric_name == DR_T_NAME:
			continue
		dr_t_vs_metric_fe, dr_t_vs_metric_p = fisher_exact([[dr_t_fp, t_genes_num],
															[fp_num, t_genes_num]])
		row = [metric_name, 
			   get_str_ratio(dr_t_fp, t_genes_num), 
			   get_str_ratio(fp_num, t_genes_num), 
			   t_genes_num, dr_t_vs_metric_fe, dr_t_vs_metric_p]

		table.append(row)

	report_csv = TABLE_FOLDER / 'dr_t_fp_analysis.csv'
	write_table_to_csv(table, report_csv)


def get_d_rt_minus_dr_t_gene_ids(db, valid_gene_ids):
	dr_t_gene_scores = get_gene_scores(db, 'drt_genes', 't_prob', valid_gene_ids=valid_gene_ids)
	first_half_num = round(len(valid_gene_ids) / 2)
	dr_t_first_half_gene_ids = set(list(sort_dict_by_values(dr_t_gene_scores, reverse=False).keys())[:first_half_num])	
	return dr_t_first_half_gene_ids


def report_gene_median_stats(metric_name, gene_ids, gene_id_to_prop_dict):
	values = []
	for gene_id in gene_ids:
		if gene_id in gene_id_to_prop_dict:
			values.append(gene_id_to_prop_dict[gene_id])
	not_found_genes = len(gene_ids) - len(values)
	median = np.median(values)
	print(metric_name, 'Median:', median)
	print(metric_name, 'Genes NOT FOUND:', not_found_genes)


def report_dip_vs_domino_overlap(db, dip_genes, domino_genes):
	# combined_score_500
	# textmining_500
	print('REPORT DIP vs DOMINO')
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


def export_d_rt_model_results(db, report_dip_vs_domino=True, include_uneecon=False, top_10=False):
	valid_gene_ids = get_common_domino_gpp_and_gevir_gene_ids(db, no_domino_train=True, no_gpp_train=True, include_uneecon=include_uneecon)

	domino_d_gene_ids = set()
	domino_genes = db.main.domino_genes.find({ 'score': { '$gt': 0.5 } })
	for domino_gene in domino_genes:
		domino_d_gene_ids.add(domino_gene['hgnc_gene_id'])

	print(len(domino_d_gene_ids))
	domino_d_gene_ids = domino_d_gene_ids & valid_gene_ids
	print(len(domino_d_gene_ids))
	# Next 3 lines are experimental to get custom number of top domino genes
	if top_10:
		top_genes_num = round(len(valid_gene_ids) / 10)
		domino_gene_scores = get_gene_scores(db, 'domino_genes', 'score', valid_gene_ids=valid_gene_ids)
		domino_d_gene_ids = set(list(sort_dict_by_values(domino_gene_scores, reverse=True).keys())[:top_genes_num])

	d_genes_num = len(domino_d_gene_ids)
	
	d_rt_valid_gene_ids = get_d_rt_minus_dr_t_gene_ids(db, valid_gene_ids)
	gpp_gene_scores = get_gene_scores(db, 'gpp_genes', 'score', valid_gene_ids=valid_gene_ids)
	#gdp_gene_scores = get_gene_scores(db, 'gdp_genes', 'score', valid_gene_ids=valid_gene_ids)
	d_rt_gene_scores = get_gene_scores(db, 'drt_genes', 'd_prob', valid_gene_ids=d_rt_valid_gene_ids)
	virlof_gene_scores = get_gene_scores(db, 'gevir_genes', 'virlof', valid_gene_ids=valid_gene_ids)
	if include_uneecon:
		uneecon_gene_scores = get_gene_scores(db, 'uneecon_genes', 'score', valid_gene_ids=valid_gene_ids)

	gpp_top_d_gene_ids = set(list(sort_dict_by_values(gpp_gene_scores, reverse=True).keys())[:d_genes_num])
	#gdp_top_d_gene_ids = set(list(sort_dict_by_values(gdp_gene_scores, reverse=True).keys())[:d_genes_num])
	dr_top_d_gene_ids = set(list(sort_dict_by_values(d_rt_gene_scores, reverse=True).keys())[:d_genes_num])
	virlof_top_d_gene_ids = set(list(sort_dict_by_values(virlof_gene_scores, reverse=False).keys())[:d_genes_num])
	if include_uneecon:
		uneecon_top_d_gene_ids = set(list(sort_dict_by_values(uneecon_gene_scores, reverse=True).keys())[:d_genes_num])

	metrics_top_t_sets = OrderedDict()
	metrics_top_t_sets[DR_T_NAME] = dr_top_d_gene_ids
	metrics_top_t_sets['GPP'] = gpp_top_d_gene_ids
	#metrics_top_t_sets['GDP'] = gdp_top_d_gene_ids
	metrics_top_t_sets['VIRLoF'] = virlof_top_d_gene_ids
	metrics_top_t_sets['DOMINO'] = domino_d_gene_ids

	if include_uneecon:
		metrics_top_t_sets['UNEECON'] = uneecon_top_d_gene_ids

	tol_num = round(len(valid_gene_ids) / 4)
	tol_gene_scores = get_gene_scores(db, 'drt_genes', 't_prob', valid_gene_ids=valid_gene_ids)
	tol_gene_ids = set(list(sort_dict_by_values(tol_gene_scores, reverse=True).keys())[:tol_num])


	print(len(dr_top_d_gene_ids & domino_d_gene_ids))

	if report_dip_vs_domino:
		report_dip_vs_domino_overlap(db, dr_top_d_gene_ids, domino_d_gene_ids)

	metric_stats = OrderedDict()
	metric_clf_perf = OrderedDict()
	gg = GeneGroups(db, valid_gene_ids=valid_gene_ids)

	gene_id_to_virlof = get_gene_id_to_virlof(db)

	print('Check Median VIRLoF for GDIT AD/AR genes')
	print('AD {}'.format(len(gg.gdit_ad_gene_ids)))
	report_gene_median_stats('VIRLoF', gg.gdit_ad_gene_ids, gene_id_to_virlof)
	print('AR {}'.format(len(gg.gdit_ar_gene_ids)))
	report_gene_median_stats('VIRLoF', gg.gdit_ar_gene_ids, gene_id_to_virlof)

	for metric_name, top_t_gene_ids in metrics_top_t_sets.items():
		metric_gene_groups = OrderedDict()
		group_set = set(gg.gdit_ad_gene_ids)
		p_all_num = len(group_set)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		tp_num = group_and_top_t_gene_num # AD
		metric_gene_groups['AD ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.gdit_ad_ar_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		fp_num = group_and_top_t_gene_num # AD,AR
		metric_gene_groups['AD,AR ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.gdit_ar_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['AR ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))		
		group_set = set(gg.cell_essential_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Cell Essential ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.cell_non_essential_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Cell Non-essential ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.olfactory_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Olfactory ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		
		# Extra sets for comparison
		group_set = set(gg.severe_hi_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Severe HI ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.gene4denovo_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Gene4Denovo ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))
		group_set = set(gg.mouse_het_lethal_gene_ids)
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['Mouse Het Lethal ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))

		# Extra-extra data
		group_set = tol_gene_ids
		group_and_top_t_gene_num = len(top_t_gene_ids & group_set)
		metric_gene_groups['DIP>75% ({})'.format(len(group_set))] = get_str_ratio(group_and_top_t_gene_num, len(group_set))

		metric_stats[metric_name] = metric_gene_groups
		metric_clf_perf[metric_name] = calculate_clf_performance(tp_num, fp_num, p_all_num)


	title = 'Table 2. Analysis of ' + get_str_ratio(d_genes_num, len(valid_gene_ids)) + \
			' genes predicted to be AD by DOMINO (score > 0.5) and ' + \
			'the same number of genes with the highest predicted probability of being AD/Disease by ' + \
			D_RT_NAME + ' and GPP or the highest VIRLoF scores' + \
			'out of ' + str(len(valid_gene_ids)) + ' genes which have predicted scores by all metrics ' + \
			'and were not used in GPP, DOMINO, ' + D_RT_NAME + ' and ' + DR_T_NAME + ' model training.'

	metric_names = list(metric_stats.keys())
	gene_group_names = metric_stats[metric_names[0]].keys()
	headers = ['Gene Group\\Metric'] + metric_names
	table = [[title], headers]
	for gene_group_name in gene_group_names:
		row = [gene_group_name]
		for metric_name in metric_names:
			row.append(metric_stats[metric_name][gene_group_name])
		table.append(row)

	if top_10:
		report_name = 'd_rt_performance_top_10.csv'
	else:
		report_name = 'd_rt_performance.csv'

	report_csv = TABLE_FOLDER / report_name
	write_table_to_csv(table, report_csv)

	metric_names = list(metric_clf_perf.keys())
	clf_perf_stat_names = metric_clf_perf[metric_names[0]].keys()
	headers = ['Clf Perf\\Metric'] + metric_names
	table = [headers]
	for clf_perf_stat_name in clf_perf_stat_names:
		row = [clf_perf_stat_name]
		for metric_name in metric_names:
			row.append(metric_clf_perf[metric_name][clf_perf_stat_name])
		table.append(row)

	if top_10:
		report_name = 'd_rt_clf_stats_top_10.csv'
	else:
		report_name = 'd_rt_clf_stats.csv'

	report_csv = TABLE_FOLDER / report_name
	write_table_to_csv(table, report_csv)


def get_str_proportion(gene_ids, gene_group):
	gene_ids = set(gene_ids)
	gene_group = set(gene_group)
	common = len(gene_ids & gene_group)
	return '{}, {:.2f}'.format(common, common * 100 / len(gene_ids))



def report_d_rt_and_dr_t_overlap(db):
	drt_genes = db.main.drt_genes.find({ 'domino_train': False, 'domino_validation': False, 'gpp_train': False })
	total_genes = drt_genes.count()
	d_pred_genes = set()
	t_pred_genes = set()
	valid_gene_ids = set()
	for drt_gene in drt_genes:
		gene_id = drt_gene['hgnc_gene_id']
		valid_gene_ids.add(gene_id)
		if drt_gene['d_prob'] >= 0.5:
			d_pred_genes.add(gene_id)
		if drt_gene['t_prob'] >= 0.5:
			t_pred_genes.add(gene_id)

	d_and_t_pred_genes = d_pred_genes & t_pred_genes
	print('Total genes', total_genes)
	print('D genes', len(d_pred_genes))
	print('T genes', len(t_pred_genes))
	print('D&T genes', len(d_and_t_pred_genes))

	print('D&T/Total', len(d_and_t_pred_genes)/total_genes)
	gg = GeneGroups(db, valid_gene_ids)

	report_gene_group_enrichment_in_the_subset('AD', d_and_t_pred_genes, valid_gene_ids, gg.gdit_ad_gene_ids)
	report_gene_group_enrichment_in_the_subset('AR', d_and_t_pred_genes, valid_gene_ids, gg.gdit_ar_gene_ids)
	report_gene_group_enrichment_in_the_subset('CNE', d_and_t_pred_genes, valid_gene_ids, gg.cell_non_essential_gene_ids)

#################
### DIP GENES ###
#################

def get_model_all_gene_scores(db, model_name, all_gene_ids):
	clf, feature_names = get_model_clf_and_feature_names(db, model_name)
	clf_perf, gene_ids, labels = get_model_data(db, model_name)

	f = Features(db)
	y_train = np.array(labels)
	X_train = np.array(f.get_values(gene_ids, feature_names))
	clf.fit(X_train, y_train)

	X = np.array(f.get_values(all_gene_ids, feature_names))

	prob_y = clf.predict_proba(X)
	prob_y = get_class_probs(prob_y, 0)

	all_gene_probs = OrderedDict()
	for i in range(0, len(all_gene_ids)):
		all_gene_probs[all_gene_ids[i]] = prob_y[i]

	# Replace probs for training genes with averaged probs from cross-validation
	clf_perf = run_cv_model(db, clf, feature_names, model_name=model_name)
	for gene_id, prob in clf_perf.mean_sample_prob.items():
		all_gene_probs[gene_id] = prob
	return all_gene_probs


def create_drt_genes(db):
	gg = GeneGroups(db)
	all_gene_ids = list(gg.valid_gene_ids)

	d_probs = get_model_all_gene_scores(db, 'd_rt', all_gene_ids)
	t_probs = get_model_all_gene_scores(db, 'dr_t', all_gene_ids)

	print(len(d_probs), len(t_probs))
	drt_genes = []
	for gene_id in all_gene_ids:
		drt_gene = OrderedDict()
		drt_gene['hgnc_gene_id'] = gene_id
		drt_gene['hgnc_gene_name'] = gg.gene_id_to_gene_name[gene_id]
		drt_gene['d_prob'] = d_probs[gene_id]
		drt_gene['t_prob'] = t_probs[gene_id]

		if gene_id in gg.domino_train_gene_ids:
			drt_gene['domino_train'] = True
		else:
			drt_gene['domino_train'] = False
		if gene_id in gg.domino_train_ad_gene_ids:
			drt_gene['domino_class'] = 'AD'
		elif gene_id in gg.domino_train_ar_gene_ids:
			drt_gene['domino_class'] = 'AR'
		else:
			drt_gene['domino_class'] = 'None'

		if gene_id in gg.domino_validation_gene_ids:
			drt_gene['domino_validation'] = True
		else:
			drt_gene['domino_validation'] = False
		if gene_id in gg.domino_validation_ad_gene_ids:
			drt_gene['domino_class'] = 'AD'
		elif gene_id in gg.domino_validation_ar_gene_ids:
			drt_gene['domino_class'] = 'AR'

		if gene_id in gg.gpp_train_gene_ids:
			drt_gene['gpp_train'] = True
		else:
			drt_gene['gpp_train'] = False
		if gene_id in gg.gpp_train_tolerant_gene_ids:
			drt_gene['gpp_class'] = 'Non-disease'
		elif gene_id in gg.gpp_train_disease_gene_ids:
			drt_gene['gpp_class'] = 'Disease'
		else:
			drt_gene['gpp_class'] = 'None'		

		drt_genes.append(drt_gene)

	db.main.drt_genes.drop()
	db.main.drt_genes.insert_many(drt_genes)
	db.main.drt_genes.create_index([('hgnc_gene_id', pymongo.ASCENDING)], name='hgnc_gene_id_1')
	db.main.drt_genes.create_index([('hgnc_gene_name', pymongo.ASCENDING)], name='hgnc_gene_name_1')


def get_sorted_genes_list(db, score_field, reverse=False, collection_name='drt_genes', ignored_genes=set(), filters={}):
	ignored_genes = set(ignored_genes)
	gene_scores = {}
	genes = db.main[collection_name].find(filters)
	for gene in genes:
		gene_id = gene['hgnc_gene_id']
		if gene_id not in ignored_genes:
			gene_scores[gene['hgnc_gene_id']] = gene[score_field]

	gene_scores = sort_dict_by_values(gene_scores, reverse=reverse)
	return gene_scores


def get_ranked_gene_enrichment_stats(gene_ids, group_gene_ids, offset_percent):
	group_gene_ids = set(group_gene_ids)
	gene_num = len(gene_ids)
	group_gene_num = len(set(gene_ids) & group_gene_ids)
	enrichment_offset = round(len(gene_ids) * (offset_percent / 100))

	gene_enrichment_stats = OrderedDict()

	bar = progressbar.ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()
	for x in range(0,gene_num):
		start = x - enrichment_offset
		stop = x + enrichment_offset
		if start <= 0:
			start = 0
			stop += 1
		elif stop >= gene_num:
			stop = gene_num
		else:
			start -= 1

		# Add 1 to start to include first gene which has 0 index in the list
		examined_range = '{:.2f}-{:.2f}%'.format(float(start + 1) * 100 / gene_num, float(stop) * 100 / gene_num)

		gr_window = set(gene_ids[start:stop])
		gr_window_num = len(gr_window)

		gene_id = gene_ids[x]

		gr_group_gene_num = len(group_gene_ids & gr_window)
		fe, p = fisher_exact([[gr_group_gene_num, gr_window_num],[group_gene_num, gene_num]])
		gene_data = OrderedDict()
		gene_data['examined_range'] = examined_range
		gene_data['fold_enrichment'] = fe
		gene_data['p_value'] = p
		gene_enrichment_stats[gene_id] = gene_data

		bar.update(x / gene_num)
	bar.finish()

	return gene_enrichment_stats


def sort_genes_based_on_virlof(db, gene_ids):
	virlof_scores = {}
	virlof_genes = db.main.gevir_genes.find({})
	for virlof_gene in virlof_genes:
		virlof_scores[virlof_gene['hgnc_gene_id']] = virlof_gene['virlof']

	not_found_genes = []
	found_genes = {}
	for gene_id in gene_ids:
		if gene_id in virlof_scores:
			found_genes[gene_id] = virlof_scores[gene_id]
		else:
			not_found_genes.append(gene_id)

	not_found_genes.sort()
	sorted_genes = sort_dict_by_values(found_genes)
	sorted_gene_ids = list(sorted_genes.keys()) + not_found_genes
	return sorted_gene_ids


def create_dip_genes(db, virlof_resort=False):
	# Note some genes have same scores and their ranking is currently not controlled
	filters = { 'domino_train': False, 'domino_validation': False, 'gpp_train': False }
	drt_genes = db.main.drt_genes.find(filters)
	mid_gene_num = round(drt_genes.count() / 2)
	#print(mid_gene_num)

	# Get last half of the ranked gene list based on 
	t_gene_scores = get_sorted_genes_list(db, 't_prob', reverse=False, filters=filters)
	t_gene_ids = list(t_gene_scores.keys())
	ranked_t_gene_ids = t_gene_ids[mid_gene_num:]
	#print(len(t_gene_ids[mid_gene_num:]))
	#print(t_gene_scores[t_gene_ids[len(t_gene_ids) - 1]])

	d_gene_scores = get_sorted_genes_list(db, 'd_prob', reverse=True, filters=filters,
											ignored_genes=ranked_t_gene_ids)

	ranked_d_gene_ids = list(d_gene_scores.keys())

	# Resort genes with AD prob >= 0.5 based on their VIRLoF scores
	if virlof_resort:
		pred_ad_genes = []
		pred_ar_genes = []

		for gene_id, prob in d_gene_scores.items():
			if prob >= 0.5:
				pred_ad_genes.append(gene_id)
			else:
				pred_ar_genes.append(gene_id)

		pred_ad_genes = sort_genes_based_on_virlof(db, pred_ad_genes)
		ranked_d_gene_ids = pred_ad_genes + pred_ar_genes

	dip_gene_ids = ranked_d_gene_ids + ranked_t_gene_ids

	dip_ranks = calculate_percentiles(dip_gene_ids, reverse=False)

	gg = GeneGroups(db, valid_gene_ids=dip_gene_ids)

	ad_set = set(gg.gdit_ad_gene_ids)
	ar_set = set(gg.gdit_ar_gene_ids)
	ad_ar_set = set(gg.gdit_ad_ar_gene_ids)
	cell_non_essential_set = set(gg.cell_non_essential_gene_ids)
	#disease_set = ad_set | ar_set | ad_ar_set

	print('Calculating AD FE stats...')
	ad_stats = get_ranked_gene_enrichment_stats(dip_gene_ids, ad_set, 5)
	print('Calculating AD+AR FE stats...')
	ad_ar_stats = get_ranked_gene_enrichment_stats(dip_gene_ids, ad_ar_set, 5)
	print('Calculating AR FE stats...')
	ar_stats = get_ranked_gene_enrichment_stats(dip_gene_ids, ar_set, 5)
	print('Cell-non-essential FE stats...')
	cell_non_essential_stats = get_ranked_gene_enrichment_stats(dip_gene_ids, cell_non_essential_set, 5)
	#print('Calculating Disease (AD, AD+AR, AR) FE stats...')
	#disease_stats = get_ranked_gene_enrichment_stats(dip_gene_ids, disease_set, 5)

	dip_genes = []
	drt_genes = db.main.drt_genes.find(filters)
	for drt_gene in drt_genes:
		gene_id = drt_gene['hgnc_gene_id']
		dip_gene = OrderedDict()
		dip_gene['hgnc_gene_id'] = gene_id
		dip_gene['hgnc_gene_name'] = drt_gene['hgnc_gene_name']
		dip_gene['d_prob'] = drt_gene['d_prob']
		dip_gene['t_prob'] = drt_gene['t_prob']
		dip_gene['dip_rank'] = dip_ranks[gene_id]
		# Examined range is based on the dip rank so is the same for all examined groups
		dip_gene['fe_examined_range'] = ad_stats[gene_id]['examined_range']
		dip_gene['ad_fe'] = ad_stats[gene_id]['fold_enrichment']
		dip_gene['ad_p'] = ad_stats[gene_id]['p_value']
		dip_gene['ad_ar_fe'] = ad_ar_stats[gene_id]['fold_enrichment']
		dip_gene['ad_ar_p'] = ad_ar_stats[gene_id]['p_value']
		dip_gene['ar_fe'] = ar_stats[gene_id]['fold_enrichment']
		dip_gene['ar_p'] = ar_stats[gene_id]['p_value']
		#dip_gene['disease_fe'] = disease_stats[gene_id]['fold_enrichment']
		#dip_gene['disease_p'] = disease_stats[gene_id]['p_value']
		dip_gene['cell_non_essential_fe'] = cell_non_essential_stats[gene_id]['fold_enrichment']
		dip_gene['cell_non_essential_p'] = cell_non_essential_stats[gene_id]['p_value']
		dip_genes.append(dip_gene)
		
	if virlof_resort:
		collection_name = 'dip_virlof_genes'
	else:
		collection_name = 'dip_genes'

	db.main[collection_name].drop()
	db.main[collection_name].insert_many(dip_genes)
	db.main[collection_name].create_index([('hgnc_gene_id', pymongo.ASCENDING)], name='hgnc_gene_id_1')
	db.main[collection_name].create_index([('hgnc_gene_name', pymongo.ASCENDING)], name='hgnc_gene_name_1')


def analyse_top_dip_results(db, rank_threshold):
	gg = GeneGroups(db)
	top_dip_gene_ids = set()
	all_dip_gene_ids = set()
	dip_genes = db.main.dip_genes.find({})
	for dip_gene in dip_genes:
		gene_id = dip_gene['hgnc_gene_id']
		all_dip_gene_ids.add(gene_id)
		if dip_gene['dip_rank'] <= rank_threshold:
			top_dip_gene_ids.add(gene_id)

	gg = GeneGroups(db, all_dip_gene_ids)
	report_gene_group_enrichment_in_the_subset('AD', top_dip_gene_ids, all_dip_gene_ids, 
											   gg.gdit_ad_gene_ids)
	report_gene_group_enrichment_in_the_subset('Severe HI', top_dip_gene_ids, all_dip_gene_ids, 
											   gg.severe_hi_gene_ids)
	report_gene_group_enrichment_in_the_subset('Gene4Denovo', top_dip_gene_ids, all_dip_gene_ids, 
											   gg.gene4denovo_gene_ids)
	report_gene_group_enrichment_in_the_subset('Mouse Het Lethal', top_dip_gene_ids, all_dip_gene_ids, 
											   gg.mouse_het_lethal_gene_ids)
	report_gene_group_enrichment_in_the_subset('Clinicaly Relevant OMIM', top_dip_gene_ids, all_dip_gene_ids, 
											   gg.gdit_omim_gene_ids)
	


def report_ad_ar_groups_stats(db):
	gene_id_to_virlof = get_gene_id_to_virlof(db)
	gg = GeneGroups(db)

	report_gene_median_stats('VIRLoF', gg.domino_train_ad_gene_ids, gene_id_to_virlof)
	report_gene_median_stats('VIRLoF', gg.domino_train_ar_gene_ids, gene_id_to_virlof)
	report_gene_median_stats('VIRLoF', gg.domino_validation_ad_gene_ids, gene_id_to_virlof)
	report_gene_median_stats('VIRLoF', gg.domino_validation_ar_gene_ids, gene_id_to_virlof)


def export_models_feature_weights(db):
	gg = GeneGroups(db)
	f = Features(db)

	# DR_T
	clf, feature_names = get_model_clf_and_feature_names(db, 'dr_t')

	y_train = gg.gpp_train_labels
	X_train = f.get_values(gg.gpp_train_gene_ids, feature_names)
	clf.fit(X_train, y_train)
	dr_t_feature_weights_list = list(clf.feature_importances_)

	dr_t_feature_weights_dict = {}
	for x in range(0, len(feature_names)):
		dr_t_feature_weights_dict[feature_names[x]] = '{:.3f}'.format(dr_t_feature_weights_list[x])

	# D_RT
	clf, feature_names = get_model_clf_and_feature_names(db, 'd_rt')

	y_train = gg.domino_train_labels
	X_train = f.get_values(gg.domino_train_gene_ids, feature_names)
	clf.fit(X_train, y_train)
	d_rt_feature_weights_list = list(clf.feature_importances_)

	d_rt_feature_weights_dict = {}
	for x in range(0, len(feature_names)):
		d_rt_feature_weights_dict[feature_names[x]] = '{:.3f}'.format(d_rt_feature_weights_list[x])

	all_feature_names = get_all_feature_names()

	headers = ['Feature name', DR_T_NAME, D_RT_NAME]
	table = [headers]
	for feature_name in all_feature_names:
		feature_data = OrderedDict()
		feature_data['Name'] = feature_name

		if feature_name in dr_t_feature_weights_dict:
			feature_data[DR_T_NAME] = dr_t_feature_weights_dict[feature_name]
		else:
			feature_data[DR_T_NAME] = 'not used'

		if feature_name in d_rt_feature_weights_dict:
			feature_data[D_RT_NAME] = d_rt_feature_weights_dict[feature_name]
		else:
			feature_data[D_RT_NAME] = 'not used'
		row = list(feature_data.values())
		table.append(row)

	report_csv = TABLE_FOLDER / 'models_feature_weights.csv'
	write_table_to_csv(table, report_csv)	


def dip_fp_analysis(db):
	gg = GeneGroups(db)
	f = Features(db)
	ar_set = set(gg.gdit_ar_gene_ids)

	dip_ad_genes = db.main.dip_genes.find({ "d_prob": { "$gte": 0.5 }, "dip_rank": { "$lte": 50.0 } })

	fp_gene_ids = []
	for dip_ad_gene in dip_ad_genes:
		gene_id = dip_ad_gene['hgnc_gene_id']
		if gene_id in ar_set:
			fp_gene_ids.append(gene_id)

	print(len(fp_gene_ids))

	gene_id_to_gevir = f.get_gene_id_to_value_dict(fp_gene_ids, 'gevir')
	gene_id_to_loeuf = f.get_gene_id_to_value_dict(fp_gene_ids, 'loeuf')
	gene_id_to_knn_n1 = f.get_gene_id_to_value_dict(fp_gene_ids, 'string_first_neighbour_ad_knn_prob')
	gene_id_to_knn_n2 = f.get_gene_id_to_value_dict(fp_gene_ids, 'string_second_neighbour_ad_knn_prob')

	knn_n1_fp = 0
	knn_n2_fp = 0

	gevir_fp = 0
	loeuf_fp = 0
	for gene_id in fp_gene_ids:
		if gene_id_to_knn_n1[gene_id] >= 0.5:
			knn_n1_fp += 1
		if gene_id_to_knn_n2[gene_id] >= 0.5:
			knn_n2_fp += 1
		if gene_id_to_gevir[gene_id] <= 25:
			gevir_fp += 1
		if gene_id_to_loeuf[gene_id] <= 25:
			loeuf_fp += 1
		'''
		print(gene_id, 
			  gene_id_to_gevir[gene_id],
			  gene_id_to_loeuf[gene_id],
			  gene_id_to_knn_n1[gene_id],
			  gene_id_to_knn_n2[gene_id],)
		'''
	print('knn_n1_fp', knn_n1_fp)
	print('knn_n2_fp', knn_n2_fp)
	print('gevir_fp', gevir_fp)
	print('loeuf_fp', loeuf_fp)


###################
### TEMP CHECKS ###
###################

def check_ranked_list(ranked_gene_ids, group_gene_ids, n_chunks):
	''' Temporary method to check the results '''
	print(len(ranked_gene_ids))
	group_gene_ids = set(group_gene_ids) & set(ranked_gene_ids)
	gene_id_chunks = np.array_split(ranked_gene_ids,n_chunks)

	group_genes_num = len(group_gene_ids)
	for gene_id_chunk in gene_id_chunks:
		chunk_group_prop = len(set(gene_id_chunk) & group_gene_ids) / group_genes_num
		print("{0:.2f}".format(chunk_group_prop))


def check_dip_severity(db):
	dip_gene_ranks = get_metric_ranked_gene_scores(db, 'dip_genes', 'dip_rank', reverse=False)
	dip_gene_ids = list(dip_gene_ranks.keys())
	gg = GeneGroups(db, valid_gene_ids=dip_gene_ids)

	group_gene_lists = OrderedDict()

	'''
	group_gene_lists['Severe HI'] = set(gg.severe_hi_gene_ids) & set(gg.gdit_ad_gene_ids)
	group_gene_lists['De Novo'] = set(gg.gene4denovo_gene_ids) & set(gg.gdit_ad_gene_ids)
	group_gene_lists['Mouse Het Lethal'] = set(gg.mouse_het_lethal_gene_ids) & set(gg.gdit_ad_gene_ids)
	'''
	#gdit_lethal_b_gene_ids
	#gdit_lethal_a_gene_ids
	#gdit_ad_ar_gene_ids
	ad_lethal_gene_ids = set(gg.gdit_lethal_a_gene_ids) | set(gg.severe_hi_gene_ids)
	ar_lethal_gene_ids = set(gg.gdit_lethal_a_gene_ids) | set(gg.severe_ar_gene_ids)
	group_gene_lists['AD Lethal'] = set(gg.gdit_ad_gene_ids) & ad_lethal_gene_ids
	group_gene_lists['AD Non-Lethal'] = set(gg.gdit_ad_gene_ids) - ad_lethal_gene_ids
	#group_gene_lists['AD,AR Lethal'] = set(gg.gdit_ad_ar_gene_ids) & lethal_gene_ids
	#group_gene_lists['AD,AR Non-Lethal'] = set(gg.gdit_ad_ar_gene_ids) - lethal_gene_ids
	group_gene_lists['AR Lethal'] = set(gg.gdit_ar_gene_ids) & ar_lethal_gene_ids
	group_gene_lists['AR Non-Lethal'] = set(gg.gdit_ar_gene_ids) - ar_lethal_gene_ids

	for group_gene_name, group_gene_ids in group_gene_lists.items():
		group_gene_ids = list(group_gene_ids)
		print(group_gene_name)
		check_ranked_list(dip_gene_ids, group_gene_ids, 10)
		print('TOTAL', len(group_gene_ids))


def export_gene_metrics(db, include_omim_original=False, use_dip_virlof=True):
	if use_dip_virlof:
		dip_collection_name = 'dip_virlof_genes'
	else:
		dip_collection_name = 'dip_genes'

	dip_gene_ranks = get_metric_ranked_gene_scores(db, dip_collection_name, 'dip_rank', reverse=False)
	dip_gene_fe_examined_ranges = get_metric_ranked_gene_scores(db, dip_collection_name, 'fe_examined_range', reverse=False)
	dip_gene_ad_fes = get_metric_ranked_gene_scores(db, dip_collection_name, 'ad_fe', reverse=False)
	dip_gene_ad_ar_fes = get_metric_ranked_gene_scores(db, dip_collection_name, 'ad_ar_fe', reverse=False)
	dip_gene_ar_fes = get_metric_ranked_gene_scores(db, dip_collection_name, 'ar_fe', reverse=False)

	domino_genes = get_metric_ranked_gene_scores(db, 'domino_genes', 'score', reverse=True)
	gpp_genes = get_metric_ranked_gene_scores(db, 'gpp_genes', 'score', reverse=True)
	virlof_genes = get_metric_ranked_gene_scores(db, 'gevir_genes', 'virlof', reverse=False)

	if include_omim_original:
		omim_genes = get_metric_ranked_gene_scores(db, 'omim_genes', 'phenotype', reverse=True)

	rows = []
	drt_genes = db.main.drt_genes.find({})
	for drt_gene in drt_genes:
		drt_gene_stats = OrderedDict()
		for k, v in drt_gene.items():
			if k != '_id':
				drt_gene_stats[k] = v
		gene_id = drt_gene['hgnc_gene_id']
		if gene_id in dip_gene_ranks:
			drt_gene_stats['dip_rank'] = dip_gene_ranks[gene_id]
			drt_gene_stats['dip_fold_enrichment_examined_range'] = dip_gene_fe_examined_ranges[gene_id]
			drt_gene_stats['dip_ad_fold_enrichment'] = dip_gene_ad_fes[gene_id]
			drt_gene_stats['dip_ad_ar_fold_enrichment'] = dip_gene_ad_ar_fes[gene_id]
			drt_gene_stats['dip_ar_fold_enrichment'] = dip_gene_ar_fes[gene_id]
		else:
			drt_gene_stats['dip_rank'] = 'NA'
			drt_gene_stats['dip_fold_enrichment_examined_range'] = 'NA'
			drt_gene_stats['dip_ad_fold_enrichment'] = 'NA'
			drt_gene_stats['dip_ad_ar_fold_enrichment'] = 'NA'
			drt_gene_stats['dip_ar_fold_enrichment'] = 'NA'
		if gene_id in domino_genes:
			drt_gene_stats['domino_ad_prob'] = domino_genes[gene_id]
		else:
			drt_gene_stats['domino_ad_prob'] = 'NA'
		if gene_id in gpp_genes:
			drt_gene_stats['gpp_disease_prob'] = gpp_genes[gene_id]
		else:
			drt_gene_stats['gpp_disease_prob'] = 'NA'
		if gene_id in virlof_genes:
			drt_gene_stats['virlof_rank'] = virlof_genes[gene_id]
		else:
			drt_gene_stats['virlof_rank'] = 'NA'

		if include_omim_original:
			if gene_id in omim_genes:
				drt_gene_stats['omim_name_mapped'] = 'Y'
				drt_gene_stats['omim_phenotype'] = omim_genes[gene_id]
			else:
				drt_gene_stats['omim_name_mapped'] = 'N'
				drt_gene_stats['omim_phenotype'] = ''

		headers = list(drt_gene_stats.keys())
		rows.append(list(drt_gene_stats.values()))

	table = [headers] + rows
	report_csv = TABLE_FOLDER / 'gene_scores.csv'
	write_table_to_csv(table, report_csv)


############
### TEMP ###
############

def check_string_ppi_preds(db):
	dip_gene_ranks = get_metric_ranked_gene_scores(db, 'dip_genes', 'dip_rank', reverse=False)
	dip_gene_ids = list(dip_gene_ranks.keys())
	gg = GeneGroups(db)
	f = Features(db)
	train_ad = set(gg.domino_train_ad_gene_ids)
	n1_preds = f.get_gene_id_to_value_dict(dip_gene_ids, 'string_first_neighbour_ad_knn_prob')
	gene_with_n_ad_ppis = set()

	string_genes = db.main.string_genes_v10.find({})
	for string_gene in string_genes:
		gene_id = string_gene['hgnc_gene_id']
		ppi_gene_ids = set(string_gene['textmining_500'])
		n_ad_ppis = len(ppi_gene_ids & train_ad)
		#print(len(ppi_gene_ids), len(train_ad), n_ad_ppis)
		if gene_id in dip_gene_ids and n_ad_ppis == 1:
			gene_with_n_ad_ppis.add(gene_id)

	pred_genes = set()
	for gene_id in gene_with_n_ad_ppis:
		if n1_preds[gene_id] >= 0.5:
			pred_genes.add(gene_id)

	filtered_pred_genes = set()
	for gene_id in pred_genes:
		if dip_gene_ranks[gene_id] < 50:
			filtered_pred_genes.add(gene_id)
	ad_set = set(gg.gdit_ad_gene_ids)
	ar_set = set(gg.gdit_ar_gene_ids)
	print(len(pred_genes))
	print('AD', len(pred_genes & ad_set))
	print('AR', len(pred_genes & ar_set))

	print(len(filtered_pred_genes))
	print('AD', len(filtered_pred_genes & ad_set))
	print('AR', len(filtered_pred_genes & ar_set))

	for gene_id in filtered_pred_genes:
		print(gg.gene_id_to_gene_name[gene_id])


def check_gene_features(db):
	dip_gene_ranks = get_metric_ranked_gene_scores(db, 'dip_genes', 'dip_rank', reverse=False)
	dip_gene_ids = list(dip_gene_ranks.keys())

	gg = GeneGroups(db, valid_gene_ids=dip_gene_ids)
	f = Features(db)

	feature_name = 'string_first_neighbour_ad_knn_prob'
	ad_vals = list(f.get_gene_id_to_value_dict(gg.gdit_test_ad_gene_ids, feature_name).values())
	ar_vals = list(f.get_gene_id_to_value_dict(gg.gdit_test_ar_gene_ids, feature_name).values())
	olf_vals = list(f.get_gene_id_to_value_dict(gg.olfactory_gene_ids, feature_name).values())

	print(len(ad_vals))
	print(np.mean(ad_vals))

	print(len(ar_vals))
	print(np.mean(ar_vals))

	print(len(olf_vals))
	print(np.mean(olf_vals))


def main():
	db = MongoDB()

	'''
	# Test DOMINO
	domino = Domino(db)
	domino.get_clf_perf_domino_train(report=True)
	domino.get_clf_perf_domino_validation(report=True)
	domino.get_clf_perf_gdit_test(report=True)
	'''

	# CROSS-VALIDATION
	#train_cv_dr_t_model(db)
	#train_cv_d_rt_model(db, report_domino=True)
	
	# TESTING
	#test_d_rt_model(db, 'domino_validation', report_domino=True)
	#test_d_rt_model(db, 'gdit_test', report_domino=True)

	# ALL GENES DRT/DIP SCORES
	#create_drt_genes(db)
	#create_dip_genes(db)
	#create_dip_genes(db, virlof_resort=True)
	
	#export_dr_t_model_results(db, include_uneecon=False) # TODO: recheck the gene number
	#export_d_rt_model_results(db, report_dip_vs_domino=False, include_uneecon=False, top_10=False)
	#report_d_rt_and_dr_t_overlap(db)
	#analyse_top_dip_results(db, 5)
	#report_ad_ar_groups_stats(db)

	#export_gene_metrics(db, include_omim_original=False, use_dip_virlof=True)
	#export_models_feature_weights(db)
	#dip_fp_analysis(db)

	#check_dip_severity(db)

	#check_string_ppi_preds(db)
	#check_gene_features(db)

if __name__ == "__main__":
	sys.exit(main())