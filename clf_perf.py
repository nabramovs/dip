import sys
import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from terminaltables import AsciiTable
from common import write_table_to_csv


def get_str_score(score):
	score = score * 100
	return "{0:.2f}".format(score)


class ClfPerf():
	def __init__(self, clf_name, class_names=('AD', 'AR')):
		self.clf_name = clf_name
		self.c0_name = class_names[0]
		self.c1_name = class_names[1]
		self.accuracy = []
		self.c0_precision = []
		self.c0_recall = []
		self.c0_f1 = []
		self.c1_precision = []
		self.c1_recall = []
		self.c1_f1 = []	
		self.sensitivity = []
		self.specificity = []
		self.roc_auc = []

		self.mean_accuracy = 0.0
		self.mean_c0_precision = 0.0
		self.mean_c0_recall = 0.0
		self.mean_c0_f1 = 0.0
		self.mean_c1_precision = 0.0
		self.mean_c1_recall = 0.0
		self.mean_c1_f1 = 0.0
		self.mean_roc_auc = 0.0

		self.mean_sensitivity = []
		self.mean_specificity = []

		self.sample_labels = {}
		self.sample_pred_true = {}
		self.sample_pred_false = {}
		self.sample_prob = {}
		self.mean_sample_prob = {}


	def update_perf_metrics(self, y_test, y_pred, y_prob=[], prob_main_class=[], sample_ids=[]):
		self.accuracy.append(accuracy_score(y_test, y_pred))
		self.c0_precision.append(precision_score(y_test, y_pred, average=None)[0])
		self.c0_recall.append(recall_score(y_test, y_pred, average=None)[0])
		self.c0_f1.append(f1_score(y_test, y_pred, average=None)[0])
		self.c1_precision.append(precision_score(y_test, y_pred, average=None)[1])
		self.c1_recall.append(recall_score(y_test, y_pred, average=None)[1])
		self.c1_f1.append(f1_score(y_test, y_pred, average=None)[1])

		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		spec = tn / float(tn + fp)
		sens = tp / float(tp + fn)

		self.specificity.append(spec)
		self.sensitivity.append(sens)

		if len(y_prob) > 0:
			self.roc_auc.append(roc_auc_score(y_test, y_prob))

		if len(sample_ids) > 0:
			for x in range(0, len(y_pred)):
				sample_id = sample_ids[x]
				y_true = y_test[x]
				y_predicted = y_pred[x]

				if sample_id not in self.sample_labels:
					self.sample_labels[sample_id] = y_true
					self.sample_pred_true[sample_id] = 0
					self.sample_pred_false[sample_id] = 0
					if len(prob_main_class) > 0:
						self.sample_prob[sample_id] = []

				if y_true == y_predicted:
					self.sample_pred_true[sample_id] += 1
				else:
					self.sample_pred_false[sample_id] += 1

				if len(prob_main_class) > 0:
					self.sample_prob[sample_id].append(prob_main_class[x])


	def calculate_mean_metrics(self):
		self.mean_accuracy = np.mean(self.accuracy)
		self.mean_c0_precision = np.mean(self.c0_precision)
		self.mean_c0_recall = np.mean(self.c0_recall)
		self.mean_c0_f1 = np.mean(self.c0_f1)
		self.mean_c1_precision = np.mean(self.c1_precision)
		self.mean_c1_recall = np.mean(self.c1_recall)
		self.mean_c1_f1 = np.mean(self.c1_f1)
		self.mean_specificity = np.mean(self.specificity)
		self.mean_sensitivity = np.mean(self.sensitivity)

		if len(self.roc_auc) > 0:
			self.mean_roc_auc = np.mean(self.roc_auc)

		if len(self.sample_prob) > 0:
			for gene_id, probs in self.sample_prob.items():
				self.mean_sample_prob[gene_id] = np.mean(probs)


	def report_performance(self):
		table_data = [
			['Metric', self.clf_name],
			['Accuracy', get_str_score(self.mean_accuracy)],
			[self.c0_name + ' Precision', get_str_score(self.mean_c0_precision)],
			[self.c0_name + ' Recall', get_str_score(self.mean_c0_recall)],
			[self.c0_name + ' F1', get_str_score(self.mean_c0_f1)],
			[self.c1_name + ' Precision', get_str_score(self.mean_c1_precision)],
			[self.c1_name + ' Recall', get_str_score(self.mean_c1_recall)],
			[self.c1_name + ' F1', get_str_score(self.mean_c1_f1)],
			['Specificity', get_str_score(self.mean_specificity)],
			['Sensitivity', get_str_score(self.mean_sensitivity)],
		]

		if self.mean_roc_auc > 0:
			table_data.append(['ROC AUC', get_str_score(self.mean_roc_auc)])

		table = AsciiTable(table_data)
		print(table.table)

	def get_dictionary(self):
		dictionary = OrderedDict()
		dictionary['Clf'] = self.clf_name
		dictionary['Accuracy'] = get_str_score(self.mean_accuracy)
		dictionary[self.c0_name + ' Precision'] = float(get_str_score(self.mean_c0_precision))
		dictionary[self.c0_name + ' Recall'] = float(get_str_score(self.mean_c0_recall))
		dictionary[self.c0_name + ' F1'] = float(get_str_score(self.mean_c0_f1))
		dictionary[self.c1_name + ' Precision'] = float(get_str_score(self.mean_c1_precision))
		dictionary[self.c1_name + ' Recall'] = float(get_str_score(self.mean_c1_recall))
		dictionary[self.c1_name + ' F1'] = float(get_str_score(self.mean_c1_f1))
		if self.mean_roc_auc > 0:
			dictionary['ROC AUC'] = float(get_str_score(self.mean_roc_auc))
		dictionary['Specificity'] = float(get_str_score(self.mean_specificity))
		dictionary['Sensitivity'] = float(get_str_score(self.mean_sensitivity))
		return dictionary


	def export_gene_predictions(self, report_csv, sample_id_to_name={}, sample_id_extra_headers=[], sample_id_extra_columns={}):
		headers = ['sample_id', 'sample_name', 'label', 'correct', 'incorrect', 'prediction_ratio'] + sample_id_extra_headers
		table = [headers]
		for sample_id, label in self.sample_labels.items():
			correct = self.sample_pred_true[sample_id]
			incorrect = self.sample_pred_false[sample_id]
			prediction_ratio = correct / float(correct + incorrect)
			sample_name = ''
			if sample_id in sample_id_to_name:
				sample_name = sample_id_to_name[sample_id]
			row = [sample_id, sample_name, label, correct, incorrect, prediction_ratio]
			if sample_id in sample_id_extra_columns:
				row += sample_id_extra_columns[sample_id]
			else:
				row += [''] * len(sample_id_extra_headers)

			table.append(row)
		write_table_to_csv(table, report_csv)


	def report_ad_short_performance(self):
		print(get_str_score(self.mean_c0_f1), ' | ',  get_str_score(self.mean_c0_precision), ' | ',  get_str_score(self.mean_c0_recall))


def calculate_diff_between_rounded_nums(n1, n2):
	return "{0:.2f}".format(float(get_str_score(n1)) - float(get_str_score(n2)))


def report_clfs_mean_metrics_comparison(clf_0, clf_1, report_csv=''):
	clf_0.calculate_mean_metrics()
	clf_1.calculate_mean_metrics()

	table_data = [
		['Metric', clf_0.clf_name, clf_1.clf_name, 'Diff'],
		['Accuracy', get_str_score(clf_0.mean_accuracy), get_str_score(clf_1.mean_accuracy), 
					 calculate_diff_between_rounded_nums(clf_0.mean_accuracy, clf_1.mean_accuracy)],

		[clf_0.c0_name + ' Precision', get_str_score(clf_0.mean_c0_precision), get_str_score(clf_1.mean_c0_precision), 
							           calculate_diff_between_rounded_nums(clf_0.mean_c0_precision, clf_1.mean_c0_precision)],

		[clf_0.c0_name + ' Recall', get_str_score(clf_0.mean_c0_recall), get_str_score(clf_1.mean_c0_recall), 
							        calculate_diff_between_rounded_nums(clf_0.mean_c0_recall, clf_1.mean_c0_recall)],

		[clf_0.c0_name + ' F1', get_str_score(clf_0.mean_c0_f1), get_str_score(clf_1.mean_c0_f1), 
						        calculate_diff_between_rounded_nums(clf_0.mean_c0_f1, clf_1.mean_c0_f1)],

		[clf_0.c1_name + ' Precision', get_str_score(clf_0.mean_c1_precision), get_str_score(clf_1.mean_c1_precision), 
								       calculate_diff_between_rounded_nums(clf_0.mean_c1_precision, clf_1.mean_c1_precision)],

		[clf_0.c1_name + ' Recall', get_str_score(clf_0.mean_c1_recall), get_str_score(clf_1.mean_c1_recall), 
							        calculate_diff_between_rounded_nums(clf_0.mean_c1_recall, clf_1.mean_c1_recall)],

		[clf_0.c1_name + ' F1', get_str_score(clf_0.mean_c1_f1), get_str_score(clf_1.mean_c1_f1), 
						        calculate_diff_between_rounded_nums(clf_0.mean_c1_f1, clf_1.mean_c1_f1)],

		['Specificity', get_str_score(clf_0.mean_specificity), get_str_score(clf_1.mean_specificity),
						calculate_diff_between_rounded_nums(clf_0.mean_specificity, clf_1.mean_specificity)],
		['Sensitivity', get_str_score(clf_0.mean_sensitivity), get_str_score(clf_1.mean_sensitivity),
						calculate_diff_between_rounded_nums(clf_0.mean_sensitivity, clf_1.mean_sensitivity)]
		]

	if clf_0.mean_roc_auc > 0 and clf_1.mean_roc_auc > 0:
		table_data.append(['ROC AUC', get_str_score(clf_0.mean_roc_auc), get_str_score(clf_1.mean_roc_auc),
									  calculate_diff_between_rounded_nums(clf_0.mean_roc_auc, clf_1.mean_roc_auc)])

	if report_csv:
		write_table_to_csv(table_data, report_csv)

	table = AsciiTable(table_data)
	print(table.table)



def report_clfs_experiment_f1_folds_comparison(clf_0, clf_1, exp_num, fold_num):
	stop_fold = (exp_num + 1) * fold_num
	start_fold = stop_fold - fold_num
	table_data = [['Fold', 
				   clf_0.clf_name + ' ' +  clf_0.c0_name + ' F1',
				   clf_1.clf_name + ' ' +  clf_0.c0_name + ' F1',
				   'Diff ' + clf_0.c0_name + ' F1',
				   clf_0.clf_name + ' ' +  clf_0.c1_name + ' F1',
				   clf_1.clf_name + ' ' +  clf_0.c1_name + ' F1',
				   'Diff ' + clf_0.c1_name + ' F1',
				 ]]

	for f in range(start_fold, stop_fold):
		row = [f, 
		       get_str_score(clf_0.c0_f1[f]), get_str_score(clf_1.c0_f1[f]), get_str_score(clf_0.c0_f1[f] - clf_1.c0_f1[f]),
			   get_str_score(clf_0.c1_f1[f]), get_str_score(clf_1.c1_f1[f]), get_str_score(clf_0.c1_f1[f] - clf_1.c1_f1[f])]
		table_data.append(row)

	table = AsciiTable(table_data)
	print('######################################### ' + str(exp_num) + ' #########################################')
	print(table.table)


def get_class_probs(multi_class_probs, class_num):
	first_class_probs = []
	for probs in multi_class_probs:
		first_class_probs.append(probs[class_num])
	return first_class_probs


class ClfResult():
	def __init__(self, random_seed, fold_num, test_gene_ids, y_pred, y_prob, y_prob_auc, y_true):
		self.random_seed = random_seed
		self.fold_num = fold_num
		self.test_gene_ids = test_gene_ids
		self.y_pred = y_pred
		self.y_prob = y_prob
		self.y_prob_auc = y_prob_auc
		self.y_true = y_true

	def get_dictionary(self):
		dictionary = OrderedDict()
		dictionary['random_seed'] = self.random_seed
		dictionary['fold_num'] = self.fold_num
		dictionary['test_gene_ids'] = list(self.test_gene_ids)
		dictionary['y_pred'] = list(self.y_pred)
		dictionary['y_prob'] = list(self.y_prob)
		dictionary['y_prob_auc'] = list(self.y_prob_auc)
		dictionary['y_true'] = list(self.y_true)
		return dictionary


def calculate_clf_performance_from_clf_results(clf_results, clf_name):
	clf_perf = ClfPerf(clf_name=clf_name)
	for clf_result in clf_results:
		clf_perf.update_perf_metrics(clf_result.y_true, clf_result.y_pred, y_prob=clf_result.y_prob_auc)	

	clf_perf.calculate_mean_metrics()
	return clf_perf