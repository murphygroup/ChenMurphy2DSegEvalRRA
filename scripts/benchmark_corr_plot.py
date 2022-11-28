import os
import pandas as pd
import numpy as np
import pickle
import bz2
from os.path import join
from scipy.sparse import csr_matrix
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.patches import Rectangle
import glob
from sklearn.metrics import auc
import matplotlib.path as mpath



if __name__ == '__main__':
	file_dir = os.path.dirname(os.getcwd())
	data_dir = join(file_dir, 'data', 'annotation', 'CODEX', 'HBM279.TQRS.775')
	output_dir = join(file_dir, 'data', 'annotation', 'CODEX')
	tile_list = ['R001_X003_Y004', 'R001_X004_Y003']
	expert_list = ['expert1', 'expert2']
	method_list = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	           'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new',
	           'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial', 'expert1', 'expert2']

	methods_abre_list = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto', 'DeepCell 0.9.0 mem',
	                'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX', 'AICS(classic)',
	                'Cellsegm', 'Voronoi', 'Expert1', 'Expert2']
	
	quality_score_list = []
	seg_score_dataframe = pd.read_csv(join(output_dir, 'seg_prime_score', 'seg_prime_score.csv'), index_col=0)
	for i in range(len(seg_score_dataframe)):
		method = seg_score_dataframe['method'][i]
		tile = seg_score_dataframe['tile'][i]
		if tile == 'tile1':
			tile_original = 'R001_X003_Y004'
		else:
			tile_original = 'R001_X004_Y003'
		f = open(join(data_dir, tile_original, 'metrics_' + method_list[methods_abre_list.index(method)] + '_v28.json'))
		metrics = json.load(f)
		quality_score_list.append(metrics['QualityScore'])
	seg_score_dataframe['quality_score'] = quality_score_list
	
	
	# for tile_original in tile_list:
	# 	for expert in expert_list:
	# 		if tile_original == 'R001_X003_Y004':
	# 			tile = 'tile1'
	# 		else:
	# 			tile = 'tile2'
	# 		seg_score_dataframe_current = seg_score_dataframe.loc[(seg_score_dataframe['tile'] == tile) & (seg_score_dataframe['expert'] == expert)]
	# 		# seg_score_dataframe_tile2_expert1 = seg_score_dataframe.loc[seg_score_dataframe['tile'] == 'tile2' & seg_score_dataframe['expert'] == 'expert1']
	# 		seg_score_r_current = pearsonr(seg_score_dataframe_current['seg_score'], seg_score_dataframe_current['quality_score'])[0]
	# 		# seg_score_r2 = pearsonr(seg_score_dataframe_tile2_expert1['seg_score'], seg_score_dataframe_tile2_expert1['quality_score'])[0]
	#
	# 		print(tile, expert, seg_score_r_current)

	seg_score_dataframe_tile1 = seg_score_dataframe.loc[(seg_score_dataframe['tile'] == 'tile1') & (seg_score_dataframe['expert'] == 'expert2')]
	seg_score_dataframe_tile2 = seg_score_dataframe.loc[(seg_score_dataframe['tile'] == 'tile2') & (seg_score_dataframe['expert'] == 'expert2')]
	seg_score_r1 = pearsonr(seg_score_dataframe_tile1['seg_score'], seg_score_dataframe_tile2['quality_score'])[0]
	seg_score_r2 = pearsonr(seg_score_dataframe_tile1['seg_score'], seg_score_dataframe_tile2['quality_score'])[0]

	print((seg_score_r1+seg_score_r2)/2)
	
	star = mpath.Path.unit_regular_star(6)

	marker = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o", '>', '<']

	seg_score_dataframe["Annotation"] = seg_score_dataframe["expert"] + ' in ' + seg_score_dataframe["tile"]
	# r = pearsonr(seg_score_dataframe.values[:,0], seg_score_dataframe.values[:,1])[0]
	#
	# r = "%.2f" % r
	# print('SEG prime r = ', r)

	fig, ax = plt.subplots()
	p = sns.scatterplot(data=seg_score_dataframe, x='seg_score', y='quality_score', hue='Annotation', style='method', markers=marker, s=60, lw=0)
	handles, labels = ax.get_legend_handles_labels()
	# labels = ['compared with expert1 in tile1', 'compared with expert2 in tile1', 'compared with expert1 in tile2', 'compared with expert2 in tile2', 'r=' + r]
	# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
	# handles.append(extra)
	# ax.legend(handles=handles, labels=labels, loc=4)
	# h, l = ax.get_legend_handles_labels()
	# l1 = ax.legend(h[:5], l[:5], loc='lower right')
	# l2 = ax.legend(h[5:], l[5:], loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	# ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first
	# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	# ax.get_legend().remove()
	ax.tick_params(axis='both', which='major', labelsize=13)
	# ax.set(ylabel=None)
	# ax.set_yticklabels([])
	ax.legend(bbox_to_anchor=(0.5, -0.15),ncol=2)
	plt.xlabel('SEG\' Score', fontsize=13)
	plt.ylabel('Quality Score', fontsize=13)

	plt.savefig(join(file_dir, 'figures', 'seg_score_corr_prime.png'), bbox_inches='tight',dpi=500)
	# plt.show()
	plt.clf()
	
	


	f1_score_dataframe = pd.read_csv(join(output_dir, 'f1_score', 'f1_score_0.3.csv'), index_col=0)

	quality_score_list = []
	for i in range(len(f1_score_dataframe)):
		method = f1_score_dataframe['method'][i]
		tile_original = f1_score_dataframe['tile'][i]
		if tile_original == 'tile1':
			tile = 'R001_X003_Y004'
		else:
			tile = 'R001_X004_Y003'
		f = open(join(data_dir, tile, 'metrics_' + method_list[methods_abre_list.index(method)] + '_v28.json'))
		metrics = json.load(f)
		quality_score_list.append(metrics['QualityScore'])
	f1_score_dataframe['quality_score'] = quality_score_list
	# print(pearsonr(f1_score_dataframe['f1_score'], f1_score_dataframe['quality_score'])[0])
	
	f1_score_dataframe_tile1 = f1_score_dataframe.loc[(f1_score_dataframe['tile'] == 'tile1') & (f1_score_dataframe['expert'] == 'expert2')]
	f1_score_dataframe_tile2 = f1_score_dataframe.loc[(f1_score_dataframe['tile'] == 'tile2') & (f1_score_dataframe['expert'] == 'expert2')]
	f1_score_r1 = pearsonr(f1_score_dataframe_tile1['f1_score'], f1_score_dataframe_tile1['quality_score'])[0]
	f1_score_r2 = pearsonr(f1_score_dataframe_tile2['f1_score'], f1_score_dataframe_tile2['quality_score'])[0]

	print((f1_score_r1+f1_score_r2)/2)

	marker = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o", '>', '<']

	f1_score_dataframe["Annotation"] = f1_score_dataframe["expert"] + ' in ' + f1_score_dataframe["tile"]

	fig, ax = plt.subplots()
	p = sns.scatterplot(data=f1_score_dataframe, x='f1_score', y='quality_score', hue='Annotation', style='method',
	                    markers=marker, s=50)
	handles, labels = ax.get_legend_handles_labels()
	# labels = ['compared with expert1 in tile1', 'compared with expert2 in tile1', 'compared with expert1 in tile2', 'compared with expert2 in tile2', 'r=' + r]
	# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
	# handles.append(extra)
	# ax.legend(handles=handles, labels=labels, loc=4)
	h, l = ax.get_legend_handles_labels()
	# l1 = ax.legend(h[:5], l[:5], loc='lower right')
	# l2 = ax.legend(h[5:], l[5:], loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	# ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first
	# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	ax.get_legend().remove()

	ax.tick_params(axis='both', which='major', labelsize=13)

	# ax.legend(bbox_to_anchor=(0.5, -0.15),ncol=2)
	plt.xlabel('F1 Score', fontsize=13)
	plt.ylabel('Quality Score', fontsize=13)
	# plt.yticks()
	plt.savefig(join(file_dir, 'figures', 'F1_corr.png'), bbox_inches='tight', dpi=500)

	# plt.show()
	plt.clf()
	
	
	

	f1_score_dir_list = sorted(glob.glob(join(output_dir, 'f1_score', 'f1_score_*.csv')))
	for f1_score_dir in f1_score_dir_list:
		f1_score_current = pd.read_csv(f1_score_dir, index_col=0)
		if f1_score_dir == f1_score_dir_list[0]:
			f1_score_dataframe = f1_score_current
		else:
			f1_score_dataframe = pd.concat([f1_score_dataframe, f1_score_current], axis=0)
	

	
	PRAUC_score_dataframe = pd.DataFrame(columns=['PRAUC', 'expert', 'method', 'tile'])
	for tile in ['tile1', 'tile2']:
		for expert in expert_list:
			for method in methods_abre_list:
				if expert != 'e' + method[1:]:
					current_f1_score = f1_score_dataframe.loc[
						(f1_score_dataframe['method'] == method) & (f1_score_dataframe['expert'] == expert) & (
									f1_score_dataframe['tile'] == tile)]
					current_precision = current_f1_score['TP'] / (current_f1_score['TP'] + current_f1_score['FP'])
					current_recall = current_f1_score['TP'] / (current_f1_score['TP'] + current_f1_score['FN'])
					pr_matrix = np.stack((current_precision, current_recall), axis=0).T
					pr_matrix_sorted = pr_matrix[(-pr_matrix[:, 0]).argsort()]
					current_PRAUC = auc(pr_matrix_sorted[:, 0], pr_matrix_sorted[:, 1])
					PRAUC_score_dataframe.loc[len(PRAUC_score_dataframe)] = [current_PRAUC, expert,
					                                                         method, tile]

	quality_score_list = []
	for i in range(len(PRAUC_score_dataframe)):
		method = PRAUC_score_dataframe['method'][i]
		tile_original = PRAUC_score_dataframe['tile'][i]
		if tile_original == 'tile1':
			tile = 'R001_X003_Y004'
		else:
			tile = 'R001_X004_Y003'
		f = open(join(data_dir, tile, 'metrics_' + method_list[methods_abre_list.index(method)] + '_v28.json'))
		metrics = json.load(f)
		quality_score_list.append(metrics['QualityScore'])
	PRAUC_score_dataframe['quality_score'] = quality_score_list
	
	PRAUC_score_dataframe_tile1 = PRAUC_score_dataframe.loc[(PRAUC_score_dataframe['tile'] == 'tile1') & (PRAUC_score_dataframe['expert'] == 'expert2')]
	PRAUC_score_dataframe_tile2 = PRAUC_score_dataframe.loc[(PRAUC_score_dataframe['tile'] == 'tile2') & (PRAUC_score_dataframe['expert'] == 'expert2')]
	PRAUC_score_r1 = pearsonr(PRAUC_score_dataframe_tile1['PRAUC'], PRAUC_score_dataframe_tile1['quality_score'])[0]
	PRAUC_score_r2 = pearsonr(PRAUC_score_dataframe_tile2['PRAUC'], PRAUC_score_dataframe_tile2['quality_score'])[0]

	print((PRAUC_score_r1+PRAUC_score_r2)/2)
	
	# print(pearsonr(PRAUC_score_dataframe['PRAUC'], PRAUC_score_dataframe['quality_score'])[0])
	
	
	
	star = mpath.Path.unit_regular_star(6)
	
	marker = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o", '>', '<']
	
	PRAUC_score_dataframe["Annotation"] = PRAUC_score_dataframe["expert"] + ' in ' + PRAUC_score_dataframe["tile"]
	
	fig, ax = plt.subplots()
	p = sns.scatterplot(data=PRAUC_score_dataframe, x='PRAUC', y='quality_score', hue='Annotation', style='method',
	                    markers=marker, s=50)
	handles, labels = ax.get_legend_handles_labels()
	# labels = ['compared with expert1 in tile1', 'compared with expert2 in tile1', 'compared with expert1 in tile2', 'compared with expert2 in tile2', 'r=' + r]
	# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
	# handles.append(extra)
	# ax.legend(handles=handles, labels=labels, loc=4)
	h, l = ax.get_legend_handles_labels()
	# l1 = ax.legend(h[:5], l[:5], loc='lower right')
	# l2 = ax.legend(h[5:], l[5:], loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	# ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first
	# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	ax.set(ylabel=None)
	ax.set_yticklabels([])
	ax.get_legend().remove()
	# ax.legend(bbox_to_anchor=(0.5, -0.15),ncol=2)
	ax.tick_params(axis='both', which='major', labelsize=13)
	
	plt.xlabel('PRAUC', fontsize=13)
	# plt.ylabel('Quality Score', fontsize=13)
	# plt.yticks()
	plt.savefig(join(file_dir, 'figures', 'PRAUC_corr.png'), bbox_inches='tight', dpi=500)
	# plt.show()
	plt.clf()