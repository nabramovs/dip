import os
import sys
import progressbar
import pymongo
import numpy as np

from features import Features
from clf_perf import ClfPerf, ClfResult, get_class_probs
from clf_perf import report_clfs_mean_metrics_comparison
from progressbar import ProgressBar, ETA, Percentage, Bar
from collections import OrderedDict
from common import MongoDB
from gene_groups import GeneGroups

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score


PROGRESS_BAR_WIDGETS = [Percentage(), ' ', Bar(), ' ', ETA()]


#################
### CONSTANTS ###
#################

CV_N_FOLDS = 10
CV_N_RUNS = 10

COMMON_FEATURE_NAMES = ['splice_acceptor',
						'gevir',
						'loeuf',
						'domino_splice_donor',
						'domino_5_prime_utr_conservation',
						'domino_mrna_half_life_gt_10h',
						'uneecon',
						'gnomad_sv_lof_sum_af',
					]

D_RT_FEATURE_NAMES = COMMON_FEATURE_NAMES + \
					['string_first_neighbour_ad_knn_prob',
					 'string_second_neighbour_ad_knn_prob', 
					 'string_direct_ppi_virlof_domino',
					]

DR_T_FEATURE_NAMES = COMMON_FEATURE_NAMES + \
					['string_direct_ppi_virlof_gpp',
					]


def get_all_feature_names():
	feature_names = []
	for feature_name in COMMON_FEATURE_NAMES:
		feature_names.append(feature_name)
	for feature_name in DR_T_FEATURE_NAMES:
		if feature_name not in feature_names:
			feature_names.append(feature_name)	
	for feature_name in D_RT_FEATURE_NAMES:
		if feature_name not in feature_names:
			feature_names.append(feature_name)	
	return feature_names


def f1_scorer(clf, X, y):
	pred_y = clf.predict(X)
	return f1_score(y, pred_y, average=None)[0]


def run_rfecv_and_save_fs_results(db, clf, model_name, clf_parameters, feature_names, X, y):
	kf = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=0)	
	selector = RFECV(clf, step=1, cv=kf, n_jobs=-1, scoring=f1_scorer)
	selector = selector.fit(X, y)

	selected_feature_names = []
	for x in range(0, len(feature_names)):
		if selector.support_[x]:
			selected_feature_names.append(feature_names[x])

	fs_results = OrderedDict()
	fs_results['model_name'] = model_name
	fs_results['clf_parameters'] = clf_parameters
	fs_results['selected_feature_names'] = selected_feature_names
	fs_results['selected_features_num'] = len(selected_feature_names)
	fs_results['f1'] = max(selector.grid_scores_)

	db.main.feature_selection_results.insert_one(fs_results)


def run_rfecv_with_clf_parameter_tunning(db, model_name):
	db.main.feature_selection_results.delete_many({'model_name': model_name})

	gg = GeneGroups(db)
	f = Features(db)	

	if model_name == 'd_rt':
		feature_names = D_RT_FEATURE_NAMES
		X = f.get_values(np.array(gg.domino_train_gene_ids), feature_names)
		y = np.array(gg.domino_train_labels)
	elif model_name == 'dr_t':
		feature_names = DR_T_FEATURE_NAMES
		X = f.get_values(np.array(gg.gpp_train_gene_ids), feature_names)
		y = np.array(gg.gpp_train_labels)

	# Tutorial:
	# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

	# Number of trees in random forest
	n_estimators_params = list(range(100,501,100))
	# Number of features to consider at every split
	max_features_params = ['auto', None]
	# Maximum number of levels in tree
	max_depth_params = [2, 4, 6]
	max_depth_params.append(None)
	# Minimum number of samples required to split a node
	min_samples_split_and_leaf_tuple_params = [(2, 1), (4, 2), (8, 4), (16, 8)]
	# Method of selecting samples for training each tree
	bootstrap_params = [True, False]

	total_rfecv_runs = len(n_estimators_params) * len(max_features_params) * len(max_depth_params) * \
					   len(min_samples_split_and_leaf_tuple_params) * len(bootstrap_params)

	print('Total requiered RFECV runs', total_rfecv_runs)

	rfecv_run_num = 0
	bar = progressbar.ProgressBar(maxval=1.0).start()
	for n_estimators in n_estimators_params:
		for max_features in max_features_params:
			for max_depth in max_depth_params:
				for bootstrap in bootstrap_params:
					for min_samples_split_and_leaf_tuple in min_samples_split_and_leaf_tuple_params:
						min_samples_split, min_samples_leaf = min_samples_split_and_leaf_tuple

						clf_parameters = OrderedDict()
						clf_parameters['n_estimators'] = n_estimators
						clf_parameters['max_features'] = max_features
						clf_parameters['max_depth'] = max_depth
						clf_parameters['bootstrap'] = bootstrap
						clf_parameters['min_samples_split'] = min_samples_split
						clf_parameters['min_samples_leaf'] = min_samples_leaf

						clf = RandomForestClassifier(n_estimators=n_estimators,
													 max_features=max_features,
													 max_depth=max_depth,
													 bootstrap=bootstrap,
													 min_samples_split=min_samples_split,
													 min_samples_leaf=min_samples_leaf,
													 n_jobs=-1, random_state=0)

						run_rfecv_and_save_fs_results(db, clf, model_name, clf_parameters, feature_names, X, y)

						rfecv_run_num += 1
						bar.update(rfecv_run_num / total_rfecv_runs)
	bar.finish()


def select_best_peforming_model_clf_parameters_and_feature_names(db, model_name, verbose=False):
	'''
	Select the model which showed the best results in the clf paramter tunning and feature 
	selection procedure. Prioritise the model which showed the best performance measured by F1,
	but if multiple models have the same performance, then select the one which has
	the smallest number of trees (estimators) as it should have the fastest perfromance.
	If again, multiple models with the same F1 performance uses the same number of trees
	then select the one which uses the smallest number of features as it should be the simplest one.
	If further filtering is required, models with other clf parameters closer to scikit-learn defaults
	are selected.
	'''
	fs_results_collection = 'feature_selection_results' #'feature_selection_results_stable'

	# Select models with best F1 performance
	filters = {'model_name': model_name}
	fs_result = db.main[fs_results_collection].find_one(filters, sort=[('f1', pymongo.DESCENDING)])
	filters['f1'] = fs_result['f1']

	if verbose:
		print('F1:', filters['f1'], 'Models:', db.main[fs_results_collection].find(filters).count())

	# Select models with minimum number of estimators (for the fast performance)
	fs_result = db.main[fs_results_collection].find_one(filters, 
		sort=[('clf_parameters.n_estimators', pymongo.ASCENDING)])
	filters['clf_parameters.n_estimators'] = fs_result['clf_parameters']['n_estimators']

	if verbose:
		print('Min n_estimators:', filters['clf_parameters.n_estimators'], 
			  'Models:', db.main[fs_results_collection].find(filters).count())

	# Select models with minimum number of features (for the lowest complexity)
	fs_result = db.main[fs_results_collection].find_one(filters, 
		sort=[('selected_features_num', pymongo.ASCENDING)])
	filters['selected_features_num'] = fs_result['selected_features_num']

	if verbose:
		print('Min features:', filters['selected_features_num'], 
			  'Models:', db.main[fs_results_collection].find(filters).count())

	# Select models with maximum tree depth,
	# i.e. the closest to default RF parameters (None/Null = any depth)
	fs_results = db.main[fs_results_collection].find(filters)
	max_depth = 0
	for fs_result in fs_results:
		fs_result_max_depth = fs_result['clf_parameters']['max_depth']
		if fs_result_max_depth == None:
			max_depth = None
			# None == the maximum possible depth, so there is no need to check other options
			break
		elif fs_result_max_depth > max_depth:
			max_depth = fs_result_max_depth

	filters['clf_parameters.max_depth'] = max_depth
	if verbose:
		print('Max depth:', filters['clf_parameters.max_depth'], 
			  'Models:', db.main[fs_results_collection].find(filters).count())

	# Select models with the smallest min_samples_leaf, 
	# i.e. the closest to default RF parameters (1)
	fs_results = db.main[fs_results_collection].find_one(filters, 
		sort=[('clf_parameters.min_samples_leaf', pymongo.ASCENDING)])
	filters['clf_parameters.min_samples_leaf'] = fs_results['clf_parameters']['min_samples_leaf']

	if verbose:
		print('Min min_samples_leaf:', filters['clf_parameters.min_samples_leaf'], 
			  'Models:', db.main[fs_results_collection].find(filters).count())

	# Select models with max_feature = auto (sqrt) parameter (default in scikit-learn),
	# but if there are no such models, then use max_feature = None (i.e. all features), 
	# which was the only alternative considered in the feature selection process
	filters['clf_parameters.max_features'] = 'auto'
	fs_results = db.main[fs_results_collection].find(filters)
	if fs_results.count() == 0:
		filters['clf_parameters.max_features'] = None

	if verbose:
		print('Max features (prioritise default: auto):', filters['clf_parameters.max_features'], 
			  'Models:', db.main[fs_results_collection].find(filters).count())

	# Select models with bootstrap = True parameter (default in scikit-learn),
	# but if there are no such models, then use bootstrap = False, 
	# which was the only alternative considered in the feature selection process
	filters['clf_parameters.bootstrap'] = True
	fs_results = db.main[fs_results_collection].find(filters)
	if fs_results.count() == 0:
		filters['clf_parameters.bootstrap'] = False

	if verbose:
		print('Boostrap (prioritise default: True):', filters['clf_parameters.max_features'], 
			  'Models:', db.main[fs_results_collection].find(filters).count())

	# Save selected model clf parameters and feature names in the separate collection
	selected_fs_result = db.main[fs_results_collection].find_one(filters)
	selected_fs_result['_id'] = model_name
	db.main.clf_parameters.delete_one({'_id': model_name})
	db.main.clf_parameters.insert_one(selected_fs_result)


def get_model_clf_and_feature_names(db, model_name):
	# Note n_jobs=1 is set to avoid slight randomization in cross-validation evaluation
	selected_fs_result = db.main.clf_parameters.find_one({'_id': model_name})
	clf_parameters = selected_fs_result['clf_parameters']
	clf = RandomForestClassifier(n_estimators=clf_parameters['n_estimators'],
								 max_features=clf_parameters['max_features'],
								 max_depth=clf_parameters['max_depth'],
								 bootstrap=clf_parameters['bootstrap'],
								 min_samples_split=clf_parameters['min_samples_split'],
								 min_samples_leaf=clf_parameters['min_samples_leaf'],
								 n_jobs=1, random_state=0)
	feature_names = selected_fs_result['selected_feature_names']
	return clf, feature_names


def main():
	db = MongoDB()
	run_rfecv_with_clf_parameter_tunning(db, 'd_rt')
	run_rfecv_with_clf_parameter_tunning(db, 'dr_t')

	select_best_peforming_model_clf_parameters_and_feature_names(db, 'd_rt', verbose=False)
	select_best_peforming_model_clf_parameters_and_feature_names(db, 'dr_t', verbose=False)

if __name__ == "__main__":
	sys.exit(main())