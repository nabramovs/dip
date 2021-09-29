import os
import sys
import pymongo
import time
import numpy as np

from scipy.stats import pearsonr
from pathlib import Path
from collections import OrderedDict
from progressbar import ProgressBar, ETA, Percentage, Bar
from gene_groups import GeneGroups
from common import MongoDB, write_table_to_csv, correct_encoding, get_str_ratio
from clf_perf import ClfPerf, ClfResult, get_class_probs
from clf_perf import calculate_clf_performance_from_clf_results
from features import add_feature

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

PROGRESS_BAR_WIDGETS = [Percentage(), ' ', Bar(), ' ', ETA()]

TABLES_FOLDER = Path('tables/')

CV_N_RUNS = 10
CV_N_FOLDS = 10

###############################
### STRING DB Configuration ###
###############################

STRING_VERSION = 10

if STRING_VERSION == 10:
	STRING_PPIS_COLLECTION = 'string_v10'
	STRING_GENES_COLLECTION = 'string_genes_v10'
	STRING_CV_FEATURES_COLLECTION = 'string_cv_features_v10'
	STRING_CV_KNN_K_PERF_COLLECTION = 'string_cv_knn_k_perf_v10'
	STRING_CV_PROBS_COLLECTION = 'string_cv_probs_v10'
	STRING_PPI_DIRECT_VIRLOF_FEATURES_COLLECTION = 'string_ppi_direct_virlof_v10'
else:
	sys.exit()

STRING_SCORE_NAMES = ['combined_score', 'experimental', 'textmining']
STRING_CONFIDENCE_THRESHOLDS = range(100,901,100)

def get_string_ppi_group_id(score_name, confidence_threshold):
	return score_name + '_' + str(confidence_threshold)

STRING_PPI_GROUP_IDS = []
for score_name in STRING_SCORE_NAMES:
	for confidence_threshold in STRING_CONFIDENCE_THRESHOLDS:
		string_ppi_group_id = get_string_ppi_group_id(score_name, confidence_threshold)
		STRING_PPI_GROUP_IDS.append(string_ppi_group_id)


##############################
### STRING Gene PPI Groups ###
##############################

def get_core_gene_ids(db, gene_list_name='hgnc'):
	gene_ids = []
	if gene_list_name == 'domino':
		genes = db.main.domino_genes.find({})
		for gene in genes:
			gene_ids.append(gene['hgnc_gene_id'])
	elif gene_list_name == 'hgnc':
		genes = db.main.hgnc_genes.find({})
		for gene in genes:
			gene_ids.append(gene['hgnc_id'])		
	return gene_ids


def create_string_genes(db, version=STRING_VERSION):
	gene_ids = set(get_core_gene_ids(db))
	gg = GeneGroups(db)

	db.main[STRING_GENES_COLLECTION].drop()

	string_genes = []
	total_lines = len(gene_ids)
	line_num = 0
	bar = ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()
	for gene_id in gene_ids:
		string_gene = OrderedDict()
		string_gene['hgnc_gene_id'] = gene_id
		string_gene['hgnc_gene_name'] = gg.gene_id_to_gene_name[gene_id]

		for string_ppi_group_id in STRING_PPI_GROUP_IDS:
			string_gene[string_ppi_group_id] = set()

		string_ppis = db.main[STRING_PPIS_COLLECTION].find({'hgnc_gene_id_1': gene_id})

		for string_ppi in string_ppis:
			gene_id_2 = string_ppi['hgnc_gene_id_2']
			for score_name in STRING_SCORE_NAMES:
				confidence_score = string_ppi[score_name]
				for confidence_threshold in STRING_CONFIDENCE_THRESHOLDS:
					if confidence_score >= confidence_threshold:
						string_ppi_group_id = get_string_ppi_group_id(score_name, confidence_threshold)
						string_gene[string_ppi_group_id].add(gene_id_2)

		for string_ppi_group_id in STRING_PPI_GROUP_IDS:
			string_gene[string_ppi_group_id] = list(string_gene[string_ppi_group_id]) #  & gene_ids
		
		string_genes.append(string_gene)

		if len(string_genes) % 1000 == 0:
			db.main[STRING_GENES_COLLECTION].insert_many(string_genes)
			string_genes = []

		line_num += 1
		bar.update((line_num + 0.0) / total_lines)
	bar.finish()

	if len(string_genes) > 0:
		db.main[STRING_GENES_COLLECTION].insert_many(string_genes)

	db.main[STRING_GENES_COLLECTION].create_index([('hgnc_gene_id', pymongo.ASCENDING)], name='hgnc_gene_id_1')
	db.main[STRING_GENES_COLLECTION].create_index([('hgnc_gene_name', pymongo.ASCENDING)], name='hgnc_gene_name_1')


############################################
### STRING CROSS-VALIDATION PPI Features ###
############################################

def get_gene_ad_ar_total_string_ppis(db, ad_gene_ids, ar_gene_ids, ppi_group_id='combined_score_500'):
	ad_gene_ids = set(ad_gene_ids)
	ar_gene_ids = set(ar_gene_ids)
	ad_ppi_num = {}
	ar_ppi_num = {}
	total_ppi_num = {}
	genes_ppis = {}

	string_genes = db.main[STRING_GENES_COLLECTION].find({})
	for string_gene in string_genes:
		gene_id = string_gene['hgnc_gene_id']
		gene_ppis = set(string_gene[ppi_group_id])
		ad_ppi_num[gene_id] = len(gene_ppis & ad_gene_ids)
		ar_ppi_num[gene_id] = len(gene_ppis & ar_gene_ids)
		total_ppi_num[gene_id] = len(gene_ppis)
		genes_ppis[gene_id] = gene_ppis

	return ad_ppi_num, ar_ppi_num, total_ppi_num, genes_ppis


def get_genes_string_n1_features(db, gene_ids, ad_gene_ids, ar_gene_ids, ppi_group_id='combined_score_500'):
	ad_ppi_num, ar_ppi_num, total_ppi_num, genes_ppis = get_gene_ad_ar_total_string_ppis(db, ad_gene_ids, ar_gene_ids, ppi_group_id=ppi_group_id)
	gene_id_to_n1_features = OrderedDict()
	for gene_id in gene_ids:
		if gene_id in total_ppi_num and total_ppi_num[gene_id] > 0:
			ad_score = float(ad_ppi_num[gene_id]) / total_ppi_num[gene_id]
			ar_score = float(ar_ppi_num[gene_id]) / total_ppi_num[gene_id]
		else:
			ad_score = 0
			ar_score = 0

		gene_id_to_n1_features[gene_id] = [ad_score, ar_score]
	return gene_id_to_n1_features


def get_genes_string_n2_features(db, gene_ids, ad_gene_ids, ar_gene_ids, ppi_group_id='combined_score_500', leave_one_out=False):
	ad_ppi_num, ar_ppi_num, total_ppi_num, genes_ppis = get_gene_ad_ar_total_string_ppis(db, ad_gene_ids, ar_gene_ids, ppi_group_id=ppi_group_id)

	gene_id_to_n2_features = OrderedDict()
	for gene_id in gene_ids:
		ad_score = 0
		ar_score = 0

		if gene_id in genes_ppis:
			for ppi_gene_id in genes_ppis[gene_id]:
				ppi_gene_ad_num = ad_ppi_num[ppi_gene_id]
				if leave_one_out and gene_id in ad_gene_ids:
					ppi_gene_ad_num -= 1

				ppi_gene_ar_num = ar_ppi_num[ppi_gene_id]
				if leave_one_out and gene_id in ar_gene_ids:
					ppi_gene_ar_num -= 1

				ppi_gene_total_num = float(total_ppi_num[ppi_gene_id])

				ppi_gene_ad_score = ppi_gene_ad_num / ppi_gene_total_num
				ppi_gene_ar_score = ppi_gene_ar_num / ppi_gene_total_num

				ad_score += ppi_gene_ad_score
				ar_score += ppi_gene_ar_score

		gene_id_to_n2_features[gene_id] = [ad_score, ar_score]
	return gene_id_to_n2_features


def get_exp_id(random_seed, fold_num):
	return str(random_seed) + '_' + str(fold_num)


def calculate_knn_cv_features(db, gene_ids, gene_labels, ad_gene_ids, ar_gene_ids, random_seeds=range(0, CV_N_RUNS), n_folds=CV_N_FOLDS, ppi_group_id='combined_score_500'):
	gene_ids = np.array(gene_ids)
	y = np.array(gene_labels)

	#start = time.time()

	n1_knn_cv_features = OrderedDict()
	n2_knn_cv_features = OrderedDict()

	total_lines = CV_N_RUNS * CV_N_FOLDS
	line_num = 0
	bar = ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()
	for random_seed in random_seeds:
		kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
		fold_num = 1
		for train, test in kf.split(gene_ids, y):
			#print(random_seed, fold_num)
			train_ad_gene_ids = set(ad_gene_ids) - set(gene_ids[test])
			train_ar_gene_ids = set(ar_gene_ids) - set(gene_ids[test])
			n1_gene_id_to_string_features = get_genes_string_n1_features(db, gene_ids, train_ad_gene_ids, train_ar_gene_ids, ppi_group_id=ppi_group_id)		
			n2_gene_id_to_string_features = get_genes_string_n2_features(db, gene_ids, train_ad_gene_ids, train_ar_gene_ids, ppi_group_id=ppi_group_id)

			exp_id = get_exp_id(random_seed, fold_num)
			n1_knn_cv_features[exp_id] = n1_gene_id_to_string_features
			n2_knn_cv_features[exp_id] = n2_gene_id_to_string_features

			fold_num += 1

			line_num += 1
			bar.update((line_num + 0.0) / total_lines)

	bar.finish()	
	#end = time.time()
	#print(end - start)

	db.main[STRING_CV_FEATURES_COLLECTION].delete_one({'_id': 'n1_knn_cv_features_' + ppi_group_id})
	db.main[STRING_CV_FEATURES_COLLECTION].delete_one({'_id': 'n2_knn_cv_features_' + ppi_group_id})

	db.main[STRING_CV_FEATURES_COLLECTION].insert_one({'_id': 'n1_knn_cv_features_' + ppi_group_id, 'data': n1_knn_cv_features})
	db.main[STRING_CV_FEATURES_COLLECTION].insert_one({'_id': 'n2_knn_cv_features_' + ppi_group_id, 'data': n2_knn_cv_features})


def calculate_train_dataset_knn_cv_features(db):
	db.main[STRING_CV_FEATURES_COLLECTION].drop()
	gg = GeneGroups(db)

	for ppi_group_id in STRING_PPI_GROUP_IDS:
		print(ppi_group_id)
		calculate_knn_cv_features(db, gg.domino_train_gene_ids,
								      gg.domino_train_labels,
								      gg.domino_train_ad_gene_ids,
									  gg.domino_train_ar_gene_ids,
									  ppi_group_id=ppi_group_id)


def get_train_dataset_knn_cv_features(db, n, gene_ids, ppi_group_id):
	if n == 1:
		knn_cv_features = db.main[STRING_CV_FEATURES_COLLECTION].find_one({'_id': 'n1_knn_cv_features_' + ppi_group_id})
	elif n == 2:
		knn_cv_features = db.main[STRING_CV_FEATURES_COLLECTION].find_one({'_id': 'n2_knn_cv_features_' + ppi_group_id})

	knn_cv_features = knn_cv_features['data']

	ordered_knn_cv_features = OrderedDict()
	for exp_id, gene_id_to_knn_features in knn_cv_features.items():
		features = []
		for gene_id in gene_ids:
			row = gene_id_to_knn_features[gene_id]
			features.append(row)

		ordered_knn_cv_features[exp_id] = features
	return ordered_knn_cv_features


###########################################
### STRING CROSS-VALIDATION PERFORAMNCE ###
###########################################

def get_exp_id(random_seed, fold_num):
	return str(random_seed) + '_' + str(fold_num)


def run_knn_cv_using_precomputed_features(k, gene_ids, gene_labels, cv_features, scale=False,
	                                      random_seeds=range(0, CV_N_RUNS), n_folds=CV_N_FOLDS):

	return run_knn_cv('', k, gene_ids, gene_labels, set(), set(), cv_features=cv_features, n=1, 
	                  random_seeds=random_seeds, n_folds=n_folds, scale=scale)
	

def run_knn_cv(db, k, gene_ids, gene_labels, ad_gene_ids, ar_gene_ids, cv_features={}, n=1, 
	           random_seeds=range(0, CV_N_RUNS), n_folds=CV_N_FOLDS, scale=False, ppi_group_id='combined_score_500'):

	clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
	gene_ids = np.array(gene_ids)
	y = np.array(gene_labels)

	clf_results = []
	for random_seed in random_seeds:
		kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
		fold_num = 1
		for train, test in kf.split(gene_ids, y):

			# Use precalculated features
			if cv_features:
				exp_id = get_exp_id(random_seed, fold_num)
				X = np.array(cv_features[exp_id])
			# Or calculate them using ad and ar gene sets
			else:
				# Hide known disease genes which are used for the testing
				train_ad_gene_ids = set(ad_gene_ids) - set(gene_ids[test])
				train_ar_gene_ids = set(ar_gene_ids) - set(gene_ids[test])

				if n == 1:
					gene_id_to_string_features = get_genes_string_n1_features(db, gene_ids, train_ad_gene_ids, train_ar_gene_ids, ppi_group_id=ppi_group_id)
				elif n == 2:
					gene_id_to_string_features = get_genes_string_n2_features(db, gene_ids, train_ad_gene_ids, train_ar_gene_ids, ppi_group_id=ppi_group_id)

				X = np.array(gene_id_to_string_features.values())

			X_train = X[train]
			X_test = X[test]

			# We found that scaling didn't really help with this data,
			# but left this option just in case
			if scale:
				sc = StandardScaler()
				X_train = sc.fit_transform(X_train)
				X_test = sc.transform(X_test)

			clf.fit(X_train, y[train])

			y_pred = clf.predict(X_test)
			y_probs = clf.predict_proba(X_test)
			y_prob = get_class_probs(y_probs, 0)
			y_prob_auc = get_class_probs(y_probs, 1)
			y_true = y[test]
			test_gene_ids = gene_ids[test]

			clf_result = ClfResult(random_seed, fold_num, test_gene_ids, y_pred, y_prob, y_prob_auc, y_true)
			clf_results.append(clf_result)

			fold_num += 1

	return clf_results


def analyse_knn_performance_with_various_k(db, n, max_k=150, clean=False, scale=False):
	if n > 2:
		print('"n" parameter must be 1 or 2, not', n)
		sys.exit()

	gg = GeneGroups(db)
	gene_ids = gg.domino_train_gene_ids
	gene_labels = gg.domino_train_labels

	total_lines = max_k * len(STRING_PPI_GROUP_IDS)
	line_num = 0
	bar = ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()

	for ppi_group_id in STRING_PPI_GROUP_IDS:
		cv_features = get_train_dataset_knn_cv_features(db, n, gg.domino_train_gene_ids, ppi_group_id)
		
		for k in range(1, max_k+1):
			line_num += 1
			bar.update((line_num + 0.0) / total_lines)

			string_cv_knn_k_perf_id = ppi_group_id + '_n_' + str(n) + '_k_' + str(k)
			if scale:
				string_cv_knn_k_perf_id += '_scaled'

			# Do not recalculate perf for parameters which were checked in the previous analysis run,
			# unless this is a "clean" run
			if not clean:
				knn_run_perf = db.main[STRING_CV_KNN_K_PERF_COLLECTION].find_one({'_id': string_cv_knn_k_perf_id})
				if knn_run_perf:
					continue

			clf_results = run_knn_cv_using_precomputed_features(k, gene_ids, gene_labels, cv_features)

			clf_name = 'N' + str(n) +'_KNN_K' + str(k)
			clf_perf = calculate_clf_performance_from_clf_results(clf_results, clf_name)
			#clf_perf.report_performance()

			knn_run_perf = OrderedDict()
			knn_run_perf['_id'] = string_cv_knn_k_perf_id
			knn_run_perf['n'] = n
			knn_run_perf['k'] = k
			knn_run_perf['ppi_group_id'] = ppi_group_id
			knn_run_perf['ad_precision'] = clf_perf.mean_c0_precision
			knn_run_perf['ad_recall'] = clf_perf.mean_c0_recall
			knn_run_perf['ad_f1'] = clf_perf.mean_c0_f1
			knn_run_perf['roc_auc'] = clf_perf.mean_roc_auc
			knn_run_perf['scale'] = scale

			db.main[STRING_CV_KNN_K_PERF_COLLECTION].insert_one(knn_run_perf)

	bar.finish()


def analyse_knn_n1_and_n2_with_various_k(db, clean=False):
	if clean:
		db.main[STRING_CV_KNN_K_PERF_COLLECTION].drop()
	analyse_knn_performance_with_various_k(db, 1, clean=clean)
	analyse_knn_performance_with_various_k(db, 2, clean=clean)


def export_knn_stats(db):
	headers = ['n', 'k', 'ad_precision', 'ad_recall', 'ad_f1', 'roc_auc', 'scale']
	table = [headers]
	knn_k_stats = db.main[STRING_CV_KNN_K_PERF_COLLECTION].find({})
	for knn_k in knn_k_stats:
		row = []
		for c_name in headers:
			row.append(knn_k[c_name])
		table.append(row)

	output_csv = TABLES_FOLDER / 'knn_k_perf.csv'
	write_table_to_csv(table, output_csv)


#######################################################
### STRING KNN N1 and N2 FINAL FEATURES (ALL GENES) ###
#######################################################

def create_string_knn_n1_and_n2_final_features(db):
	for n in [1, 2]:
		best_params = get_best_n1_or_n2_knn_parameters(db, n)
		calculate_string_knn_feature(db, best_params['ppi_group_id'], best_params['n'],
			                         best_params['k'], clean=True, scale=False)


def get_best_n1_or_n2_knn_parameters(db, n, string_ppi='', auc=False, report=False):
	if string_ppi:
		filters = {"n": n, "ppi_group_id": string_ppi}
	else:
		filters = {"n": n}
	perf_stat = 'ad_f1'
	if auc:
		perf_stat = 'roc_auc'

	best_results = db.main[STRING_CV_KNN_K_PERF_COLLECTION].find_one(filters, sort=[(perf_stat, pymongo.DESCENDING)])

	if report:
		for key, value in best_results.items():
			print(key, value)

	return best_results


def calculate_string_knn_feature(db, ppi_group_id, n, k, clean=False, scale=False):
	gg = GeneGroups(db)
	gene_ids = gg.domino_train_gene_ids
	gene_labels = gg.domino_train_labels

	knn_k_stats_feature_data_id = ppi_group_id + '_n_' + str(n) + '_k_' + str(k)
	
	clf_test_data = db.main[STRING_CV_PROBS_COLLECTION].find_one({'_id': knn_k_stats_feature_data_id})

	if not clf_test_data or clean:
		cv_features = get_train_dataset_knn_cv_features(db, n, gg.domino_train_gene_ids, ppi_group_id)
		clf_results = run_knn_cv_using_precomputed_features(k, gene_ids, gene_labels, cv_features)

		clf_perf = calculate_clf_performance_from_clf_results(clf_results, knn_k_stats_feature_data_id)
		#clf_perf.report_performance()

		knn_run_stats = OrderedDict()
		knn_run_stats['_id'] = knn_k_stats_feature_data_id
		knn_run_stats['n'] = n
		knn_run_stats['k'] = k
		knn_run_stats['string_ppi_group_id'] = ppi_group_id
		knn_run_stats['ad_precision'] = clf_perf.mean_c0_precision
		knn_run_stats['ad_recall'] = clf_perf.mean_c0_recall
		knn_run_stats['ad_f1'] = clf_perf.mean_c0_f1
		knn_run_stats['roc_auc'] = clf_perf.mean_roc_auc
		knn_run_stats['scale'] = scale

		clf_test_data = []
		for clf_result in clf_results:
			clf_test_data.append(clf_result.get_dictionary())

		knn_run_stats['clf_test_data'] = clf_test_data

		knn_run_stats = correct_encoding(knn_run_stats)
		db.main[STRING_CV_PROBS_COLLECTION].delete_one({'_id': knn_k_stats_feature_data_id})
		db.main[STRING_CV_PROBS_COLLECTION].insert_one(knn_run_stats)
	else:
		clf_test_data = clf_test_data['clf_test_data']

	data = get_string_knn_n1_or_n2_features(db, n, k, clf_test_data, scale=scale)
	if n == 1:
		feature_id = 'string_first_neighbour_ad_knn_prob'
	elif n == 2:
		feature_id = 'string_second_neighbour_ad_knn_prob'
	default_type = 'median'
	default_value = np.median(list(data.values()))
	stats = OrderedDict()
	stats['n'] = n
	stats['k'] = k
	stats['ppi_group_id'] = ppi_group_id
	stats['knn_perf_id'] = knn_k_stats_feature_data_id
	stats['scale'] = scale
	add_feature(db, feature_id, default_type, default_value, data, stats=stats)


def calculate_mean_cv_domino_train_gene_probs(gg, clf_test_data):
	# Calculate Average AD probabilities from 10x10 Cross-Validation results
	cv_train_genes_ad_probs = {}
	for gene_id in gg.domino_train_gene_ids:
		cv_train_genes_ad_probs[gene_id] = []

	for fold_data in clf_test_data:
		fold_gene_ids = fold_data['test_gene_ids']
		fold_gene_ad_probs = fold_data['y_prob']

		for i in range(0, len(fold_gene_ids)):
			gene_id = fold_gene_ids[i]
			ad_prob = fold_gene_ad_probs[i]
			cv_train_genes_ad_probs[gene_id].append(ad_prob)

	cv_train_genes_mean_ad_probs = {}
	for gene_id, ad_probs in cv_train_genes_ad_probs.items():
		cv_train_genes_mean_ad_probs[gene_id] = float(np.mean(ad_probs))
	return cv_train_genes_mean_ad_probs


def get_string_knn_n1_or_n2_features(db, n, k, clf_test_data, scale=False):
	gg = GeneGroups(db)
	cv_train_genes_mean_ad_probs = calculate_mean_cv_domino_train_gene_probs(gg, clf_test_data)

	# Calculate KNN predictions for all core (DOMINO) genes
	gene_ids = get_core_gene_ids(db, gene_list_name='domino')

	gene_ad_probs = run_knn_and_get_genes_ad_probs(db, k, gene_ids, gg.domino_train_gene_ids,
												   gg.domino_train_labels, gg.domino_train_ad_gene_ids,
												   gg.domino_train_ar_gene_ids, n=n, scale=scale)

	# Replace KNN predictions of training genes with average Cross-Validation predictions
	for cv_train_gene_id, cv_train_ad_prob in cv_train_genes_mean_ad_probs.items():
		gene_ad_probs[cv_train_gene_id] = cv_train_ad_prob

	return gene_ad_probs


def run_knn_and_get_genes_ad_probs(db, k, gene_ids, train_gene_ids, train_gene_labels, 
	                               ad_gene_ids, ar_gene_ids, n=1, scale=False):
	if n == 1:
		gene_id_to_string_features = get_genes_string_n1_features(db, gene_ids, ad_gene_ids, ar_gene_ids)
	elif n == 2:
		gene_id_to_string_features = get_genes_string_n2_features(db, gene_ids, ad_gene_ids, ar_gene_ids)

	X = np.array(list(gene_id_to_string_features.values()))
	X_train = []
	for train_gene_id in train_gene_ids:
		X_train.append(gene_id_to_string_features[train_gene_id])
	y_train = train_gene_labels

	clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
	clf.fit(X_train, y_train)

	y_probs = clf.predict_proba(X)
	ad_probs = get_class_probs(y_probs, 0)

	string_knn_ad_probs = OrderedDict()

	for i in range(0, len(gene_ids)):
		gene_id = gene_ids[i]
		string_knn_ad_probs[gene_id] = ad_probs[i]

	return string_knn_ad_probs


########################################
### STRING DIRECT VIRLOF PPI FEATURE ###
########################################

def get_gene_id_to_virlof_dict(db, gene_ids):
	gevir_genes = db.main.gevir_genes.find({})
	gene_scores = {}

	for gevir_gene in gevir_genes:
		gene_scores[gevir_gene['hgnc_gene_id']] = gevir_gene['virlof']

	median_score = np.median(list(gene_scores.values()))
	for gene_id in gene_ids:
		if gene_id not in gene_scores:
			gene_scores[gene_id] = median_score

	return gene_scores


def create_genes_ppi_direct_virlof_features(db):
	gene_ids = get_core_gene_ids(db)
	gene_scores = get_gene_id_to_virlof_dict(db, gene_ids)

	virlof_thresholds = range(5, 101, 5)

	ppi_direct_virlof_features = {}
	# Initialise feature dicts
	for score_name in STRING_SCORE_NAMES:
		for confidence_threshold in STRING_CONFIDENCE_THRESHOLDS:
			for virlof_threshold in virlof_thresholds:
				ppi_direct_virlof_feature_id = get_string_ppi_group_id(score_name, confidence_threshold) + \
				                               '_virlof_' + str(virlof_threshold)
				ppi_direct_virlof_features[ppi_direct_virlof_feature_id] = OrderedDict()

	# string_ppi_virlof  
	string_genes = db.main[STRING_GENES_COLLECTION].find({})
	#total_lines = db.main[STRING_GENES_COLLECTION].count_documents({})
	total_lines = db.main[STRING_GENES_COLLECTION].count()
	line_num = 0
	bar = ProgressBar(maxval=1.0, widgets=PROGRESS_BAR_WIDGETS).start()
	for string_gene in string_genes:
		gene_id = string_gene['hgnc_gene_id']
		for score_name in STRING_SCORE_NAMES:
			for confidence_threshold in STRING_CONFIDENCE_THRESHOLDS:
				ppi_group_id = get_string_ppi_group_id(score_name, confidence_threshold)
				for virlof_threshold in virlof_thresholds:
					ppi_direct_virlof_feature_id = ppi_group_id + '_virlof_' + str(virlof_threshold)
					gene_ppis = set(string_gene[ppi_group_id])

					ppi_direct_virlof_count = 0
					for ppi_gene_id in gene_ppis:
						if ppi_gene_id in gene_scores:
							if gene_scores[ppi_gene_id] <= virlof_threshold:
								ppi_direct_virlof_count += 1

					ppi_direct_virlof_features[ppi_direct_virlof_feature_id][gene_id] = ppi_direct_virlof_count

		line_num += 1
		bar.update((line_num + 0.0) / total_lines)
	bar.finish()

	documents = []
	for ppi_direct_virlof_feature_id, gene_direct_virlof_ppis in ppi_direct_virlof_features.items():
		document = OrderedDict()
		document['_id'] = ppi_direct_virlof_feature_id
		document['gene_direct_virlof_ppis'] = gene_direct_virlof_ppis
		documents.append(document)

	db.main[STRING_PPI_DIRECT_VIRLOF_FEATURES_COLLECTION].drop()
	db.main[STRING_PPI_DIRECT_VIRLOF_FEATURES_COLLECTION].insert_many(documents)


def add_feature_score_to_gene_direct_virlof_ppis(db, gene_ids, labels, dataset_name):
	gene_features = OrderedDict()
	for gene_id in gene_ids:
		gene_features[gene_id] = []

	features = db.main[STRING_PPI_DIRECT_VIRLOF_FEATURES_COLLECTION].find({})
	feature_names = []
	for feature in features:
		feature_names.append(feature['_id'])
		gene_direct_virlof_ppis = feature['gene_direct_virlof_ppis']
		for gene_id in gene_ids:
			gene_features[gene_id].append(gene_direct_virlof_ppis[gene_id])

	X = []
	for gene_id, features in gene_features.items():
		X.append(features)
	X = np.array(X)
	y = labels

	#f_classif
	#mutual_info_classif
	# This will set constant random_state for mutual_info_classif to have reproducible results
	np.random.seed(0)

	feature_selector = SelectKBest(mutual_info_classif, k='all')
	feature_selector.fit(X, y)

	feature_scores = list(feature_selector.scores_)

	for i in range(0, len(feature_names)):
		feature_name = feature_names[i]
		feature_score = feature_scores[i]

		db.main[STRING_PPI_DIRECT_VIRLOF_FEATURES_COLLECTION].update_one({'_id': feature_name}, 
			                                                             { '$set': {dataset_name: feature_score} })


def add_domino_and_gpp_datasets_feature_scores_to_gene_direct_virlof_ppis(db):
	valid_gene_ids = set(get_core_gene_ids(db))
	gg = GeneGroups(db, valid_gene_ids=valid_gene_ids)

	gene_ids = gg.domino_train_gene_ids
	labels = gg.domino_train_labels
	dataset_name = 'domino_train'
	add_feature_score_to_gene_direct_virlof_ppis(db, gene_ids, labels, dataset_name)

	gene_ids = gg.gpp_train_gene_ids
	labels = gg.gpp_train_labels
	dataset_name = 'gpp_train'
	add_feature_score_to_gene_direct_virlof_ppis(db, gene_ids, labels, dataset_name)


def add_best_direct_virlof_ppis_feature(db, dataset_name, feature_name):
	best_results = db.main[STRING_PPI_DIRECT_VIRLOF_FEATURES_COLLECTION].find_one({}, sort=[(dataset_name, pymongo.DESCENDING)])
	stats = OrderedDict()
	stats['score'] = best_results[dataset_name]
	stats['string_params'] = best_results['_id']
	add_feature(db, feature_name, 'no_data', 0, best_results['gene_direct_virlof_ppis'], stats=stats)


def export_best_domino_and_gpp_gene_direct_virlof_ppis_features(db):
	add_best_direct_virlof_ppis_feature(db, 'domino_train', 'string_direct_ppi_virlof_domino')
	add_best_direct_virlof_ppis_feature(db, 'gpp_train', 'string_direct_ppi_virlof_gpp')

###############
### REPORTS ###
###############

def analyse_domino_direct_ad_method_performance(db):
	gg = GeneGroups(db)
	
	ad_set = set(gg.domino_train_ad_gene_ids)
	ar_set = set(gg.domino_train_ar_gene_ids)
	domino_train_gene_ids = set(gg.domino_train_gene_ids)

	gene_ad_ppi_nums = {}
	for gene_id in domino_train_gene_ids:
		string_gene = db.main.string_genes_v10.find_one({'hgnc_gene_id': gene_id})
		if gene_id in domino_train_gene_ids:
			gene_ad_ppi_nums[gene_id] = len(ad_set & set(string_gene['combined_score_500']))

	#print(len(gene_ad_ppi_nums), len(domino_train_gene_ids))

	'''
	# VIRLoF interactions
	gene_ad_ppi_nums = {}
	gene_virlof_direct_ppis = db.main.features.find_one({'_id': 'string_direct_ppi_virlof_domino'})
	gene_virlof_direct_ppis = gene_virlof_direct_ppis['data']
	for gene_id in domino_train_gene_ids:
		gene_ad_ppi_nums[gene_id] = gene_virlof_direct_ppis[gene_id]
	'''
	print('DOMINO direct AD interactions feature best theoretical performance.')

	ad_num = len(ad_set)

	best_ad_threshold = 0
	best_ad_precision = 0
	best_ad_recall = 0
	best_ad_f1 = 0
	best_tp = 0
	best_fp = 0

	ad_thresholds = range(0, 21)
	for ad_threshold in ad_thresholds:
		pred_ad = set()
		pred_ar = set()

		for gene_id, ad_ppi_num in gene_ad_ppi_nums.items():
			if ad_ppi_num >= ad_threshold:
				pred_ad.add(gene_id)
			else:
				pred_ar.add(gene_id)

		tp = len(pred_ad & ad_set)
		fp = len(pred_ad & ar_set)
		fn = len(pred_ar & ad_set)

		ad_precision = tp / (tp + fp)
		ad_recall = tp / (tp + fn)
		ad_f1 = 2 * (ad_precision * ad_recall) / (ad_precision + ad_recall)

		#print(ad_f1, ad_precision, ad_recall)
		if ad_f1 > best_ad_f1:
			best_ad_threshold = ad_threshold
			best_ad_precision = ad_precision
			best_ad_recall = ad_recall
			best_ad_f1 = ad_f1

			best_tp = tp
			best_fp = fp

	best_ad_precision = "{:.2f}".format(best_ad_precision * 100)
	best_ad_recall = "{:.2f}".format(best_ad_recall * 100)
	best_ad_f1 = "{:.2f}".format(best_ad_f1 * 100)

	print('AD Num', ad_num)
	print('AD Threshold >=', best_ad_threshold)
	print('AD Precision', best_ad_precision)
	print('AD Recall', best_ad_recall)
	print('AD F1', best_ad_f1)
	print('AD TP', best_tp)
	print('AD FP', best_fp)


def get_clf_test_data(db, clf_test_data_id):
	string_cv_probs = db.main.string_cv_probs_v10.find_one({'_id': clf_test_data_id})
	return string_cv_probs['clf_test_data']


def report_knn_model_results(knn_k_perf):
	print('String PPIs Type:', knn_k_perf['_id'])
	print('N:', knn_k_perf['n'])
	print('KNN K:', knn_k_perf['k'])
	print('Precision:', "{:.2f}".format(knn_k_perf['ad_precision'] * 100))
	print('Recall:', "{:.2f}".format(knn_k_perf['ad_recall'] * 100))
	print('F1:', "{:.2f}".format(knn_k_perf['ad_f1'] * 100))


def analyse_string_knn_method_performance(db):
	print('### BEST KNN COMBINED 500 RESULTS ###')
	best_combined_500_n1 = get_best_n1_or_n2_knn_parameters(db, 1, string_ppi='combined_score_500')
	report_knn_model_results(best_combined_500_n1)
	print('-------------------------')
	best_combined_500_n2 = get_best_n1_or_n2_knn_parameters(db, 2, string_ppi='combined_score_500')
	report_knn_model_results(best_combined_500_n2)
	print('-------------------------')
	print('### BEST KNN RESULTS ###')
	best_any_n1 = get_best_n1_or_n2_knn_parameters(db, 1)
	report_knn_model_results(best_any_n1)
	print('-------------------------')
	best_any_n2 = get_best_n1_or_n2_knn_parameters(db, 2)
	report_knn_model_results(best_any_n2)
	print('-------------------------')

	best_any_n1_clf_test_data = get_clf_test_data(db, best_any_n1['_id'])
	best_any_n2_clf_test_data = get_clf_test_data(db, best_any_n2['_id'])

	gg = GeneGroups(db)
	ad_set = gg.domino_train_ad_gene_ids
	n1_ad_probs = calculate_mean_cv_domino_train_gene_probs(gg, best_any_n1_clf_test_data)
	n2_ad_probs = calculate_mean_cv_domino_train_gene_probs(gg, best_any_n2_clf_test_data)

	n1_probs = []
	n2_probs = []

	for gene_id, n1_ad_prob in n1_ad_probs.items():
		n1_probs.append(n1_ad_prob)
		n2_probs.append(n2_ad_probs[gene_id])

	corr, p_value = pearsonr(n1_probs, n2_probs)
	print('Best N1 and N2 KNN models AD prob Pearson correlation: {}, (p-value: {})'.format(corr, p_value))

	# Analyse True Positive and False positive overlaps
	n1_tp = set()
	n2_tp = set()

	n1_fp = set()
	n2_fp = set()

	for gene_id, ad_prob in n1_ad_probs.items():
		if ad_prob >= 0.5:
			if gene_id in ad_set:
				n1_tp.add(gene_id)
			else:
				n1_fp.add(gene_id)

	for gene_id, ad_prob in n2_ad_probs.items():
		if ad_prob >= 0.5:
			if gene_id in ad_set:
				n2_tp.add(gene_id)
			else:
				n2_fp.add(gene_id)

	print('DOMINO Train total AD:', len(ad_set))
	print('N1 TP:', len(n1_tp), 'N1 FP:', len(n1_fp))
	print('N2 TP:', len(n2_tp), 'N2 FP:', len(n2_fp))

	
	print('TP predicted by N1 or N2:', len(n1_tp | n2_tp))
	print('TP N1 and N2 overlap:', get_str_ratio(len(n1_tp & n2_tp), len(n1_tp | n2_tp)))

	print('FP predicted by N1 or N2:', len(n1_fp | n2_fp))
	print('FP N1 and N2 overlap:', get_str_ratio(len(n1_fp & n2_fp), len(n1_fp | n2_fp)))


def main():
	db = MongoDB()

	#create_string_genes(db)

	# Calculate N1 and N2 features
	#calculate_train_dataset_knn_cv_features(db)
	#analyse_knn_n1_and_n2_with_various_k(db, clean=True)
	# Adds scores to features collection
	#create_string_knn_n1_and_n2_final_features(db)

	# Calculate VIRLoF direct ppi
	#create_genes_ppi_direct_virlof_features(db)
	#add_domino_and_gpp_datasets_feature_scores_to_gene_direct_virlof_ppis(db)
	# Adds scores to features collection
	#export_best_domino_and_gpp_gene_direct_virlof_ppis_features(db)

	# Reports
	analyse_domino_direct_ad_method_performance(db)
	analyse_string_knn_method_performance(db)


if __name__ == "__main__":
	sys.exit(main())