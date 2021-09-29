import os
import sys
import progressbar
import pymongo
import numpy as np
from scipy.stats import fisher_exact
from collections import OrderedDict
from common import MongoDB, sort_dict_by_values, calculate_percentiles
from common import get_str_ratio
from gene_groups import GeneGroups
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

FIGURES_FOLDER = Path('figures/')
SUBPLOT_LETTERS = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i'}

C_BLUE = '#0072b2'
C_ORANGE = '#e69d00' 
C_GREEN = '#009e73'
C_RED = '#f35001'
C_GRAY = '#DCDCDC'
C_BLACK = '#000000'

COLOR_PALETTE = OrderedDict()
COLOR_PALETTE['B'] = '#0072b2'  # Blue; Alternative: '#1f78b4'
COLOR_PALETTE['O'] = '#e69d00'  # Orange;  Alternative: '#ff7f00' 
COLOR_PALETTE['G'] = '#009e73'  # Green; Alternative: '#33a02c'
COLOR_PALETTE['R'] = '#e31a1c'  # Red
COLOR_PALETTE['P'] = '#6a3d9a'  # Purple
COLOR_PALETTE['Y'] = '#ffff99'  # Yellow
COLOR_PALETTE['PI'] = '#ff66cc' # Pink
COLOR_PALETTE['BR'] = '#b15928' # Brown;  Alternative: a05d2c
COLOR_PALETTE['DP'] = '#581845'

COLOR_PALETTE['BL'] = '#a6cee3' # Blue Light
COLOR_PALETTE['OL'] = '#fdbf6f' # Orange Light
COLOR_PALETTE['GL'] = '#b2df8a' # Green Light
COLOR_PALETTE['RL'] = '#ff7374' # Red Light
COLOR_PALETTE['PL'] = '#cab2d6' # Purple Light

COLOR_PALETTE['RD'] = '#a02c2c' # Red Dark

# Color blind friendly palette
COLOR_PALETTE['SB'] = '#56b3e9' # Sky Blue
COLOR_PALETTE['Y'] = '#f0e442' # Yellow
COLOR_PALETTE['V'] = '#d55e00' # Vermillion
COLOR_PALETTE['RP'] = '#cc79a7' # Reddish purple


SCORE_COLORS = OrderedDict()

# Gene Score names used in figure legends.
MY_NAME = 'DIP'
GPP_NAME = 'GPP'
DOMINO_NAME = 'DOMINO'
VIRLOF_NAME = 'VIRLoF'

SCORE_COLORS[MY_NAME] = COLOR_PALETTE['O']
SCORE_COLORS[DOMINO_NAME] = COLOR_PALETTE['B']
SCORE_COLORS[GPP_NAME] = COLOR_PALETTE['G']
SCORE_COLORS[VIRLOF_NAME] = COLOR_PALETTE['P']


######################
### Paper Figure 2 ###
######################
from common import calculate_percentiles
from evaluation import get_evaluation_common_gene_ids
from evaluation import get_metrics_dict

def draw_metric_fold_enrichment_subplot(db, subplot_num, metric_name,
										gene_xs, gene_group_ys, gene_group_colours):
	font_size = 7
	ax = plt.subplot(2,2,subplot_num)

	#ax.set_title(metric_name, loc='center', fontsize=font_size + 3)
	ax.text(40, 5.8, metric_name, va='center', fontsize=10, fontweight='bold')

	ax.text(-0.13, 1.05, SUBPLOT_LETTERS[subplot_num], transform=ax.transAxes, fontsize=7, fontweight='bold', va='top', ha='right')

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.plot([0,100], [1,1], '--', label='Expected', color=C_GRAY)

	for group_name, group_ys in gene_group_ys.items():
		plt.scatter(gene_xs, group_ys, s=1, color=gene_group_colours[group_name], label=group_name)

	plt.ylabel('Fold Enrichment', fontsize=font_size)
	plt.xlabel('Rank (%)', fontsize=font_size)

	patches = OrderedDict()

	for group_name, group_colour in gene_group_colours.items():
		patches[group_name] = Line2D([0], [0], color=group_colour)

	patches['Expected'] = Line2D([0], [0], linestyle='--', color=C_GRAY)

	bbox_to_anchor=(0.55, 0.9)


	l = plt.legend(patches.values(), patches.keys(), loc='upper center', frameon=False, 
				   fontsize=font_size, handlelength=2, bbox_to_anchor=bbox_to_anchor)

	for line in l.get_lines():
		line.set_linewidth(2.0)

	xs = list(range(0, 110, 10))
	ys = list(range(0, 7))
	plt.xticks(xs, fontsize=font_size)
	plt.yticks(ys, fontsize=font_size)
	ax.set_xticklabels([str(n) for n in range(0, 110, 10)])


def get_metrics_eval_genes_fe_data(db):
	metric_names = [MY_NAME, VIRLOF_NAME, DOMINO_NAME, GPP_NAME]
	fe_gene_groups = ['ad_fe', 'ad_ar_fe', 'ar_fe', 'cell_non_essential_fe']
	eval_genes = db.main.common_eval_genes_fe.find({})
	eval_gene_fe_data = {}
	for eval_gene in eval_genes:
		gene_id = eval_gene['hgnc_gene_id']
		eval_gene_fe_data[gene_id] = OrderedDict()
		for metric_name in metric_names:
			eval_gene_fe_data[gene_id][metric_name] = OrderedDict()
			for fe_gene_group in fe_gene_groups:
				eval_gene_fe_data[gene_id][metric_name][fe_gene_group] = eval_gene[metric_name][fe_gene_group]
	return eval_gene_fe_data


def get_metric_figure_data(metric_name, gene_ids, fe_data):
	gene_percentiles = calculate_percentiles(gene_ids)
	gene_xs = list(list(gene_percentiles.values()))
	gene_group_ys = OrderedDict()
	gene_group_ys['AD'] = []
	gene_group_ys['AD&AR'] = []
	gene_group_ys['AR'] = []
	gene_group_ys['Cell non-essential'] = []

	for gene_id in gene_ids:
		gene_group_ys['AD'].append(fe_data[gene_id][metric_name]['ad_fe'])
		gene_group_ys['AD&AR'].append(fe_data[gene_id][metric_name]['ad_ar_fe'])
		gene_group_ys['AR'].append(fe_data[gene_id][metric_name]['ar_fe'])
		gene_group_ys['Cell non-essential'].append(fe_data[gene_id][metric_name]['cell_non_essential_fe'])
	return gene_xs, gene_group_ys


def draw_fold_enrichment_figure(db):
	common_gene_ids = get_evaluation_common_gene_ids(db)
	metrics_dict = get_metrics_dict(db, valid_gene_ids=common_gene_ids, use_dip_virlof=True)
	gene_fe_data = get_metrics_eval_genes_fe_data(db)

	gene_group_colours = OrderedDict()
	gene_group_colours['AD'] = COLOR_PALETTE['R']
	gene_group_colours['AD&AR'] = COLOR_PALETTE['O']
	gene_group_colours['AR'] = COLOR_PALETTE['B']
	gene_group_colours['Cell non-essential'] = COLOR_PALETTE['G']	

	fig = plt.figure(figsize = (2,2))
	subplot_num = 1

	metrics_order = [MY_NAME, VIRLOF_NAME, DOMINO_NAME, GPP_NAME]
	for metric_name in metrics_order:
		metric_gene_ids = list(metrics_dict[metric_name])
		gene_xs, gene_group_ys = get_metric_figure_data(metric_name, metric_gene_ids, gene_fe_data)
		draw_metric_fold_enrichment_subplot(db, subplot_num, metric_name, 
			                                gene_xs, gene_group_ys, gene_group_colours)
		subplot_num += 1

	fig.set_size_inches(7, 7)
	plt.tight_layout(rect=[0, 0, 1, 1], w_pad=0.1)
	plt.savefig(FIGURES_FOLDER / 'fe_figure.png', format='png', dpi=150)
	plt.close()



def main():
	db = MongoDB()
	
	draw_fold_enrichment_figure(db)


if __name__ == "__main__":
	sys.exit(main())