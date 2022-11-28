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

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
					  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def check_match_iou(ref_arr, que_arr, j_thre):
	a = set((tuple(i) for i in ref_arr))
	b = set((tuple(i) for i in que_arr))
	match_pixel_num = len(list(a & b))
	union_pixel_num = len(list(a | b))
	j_idx = match_pixel_num / union_pixel_num
	if j_idx > j_thre:
		return True
	else:
		return False

def check_match_di(ref_arr, que_arr, j_thre):
	a = set((tuple(i) for i in ref_arr))
	b = set((tuple(i) for i in que_arr))
	match_pixel_num = len(list(a & b))
	total_pixel_num = len(list(a)) + len(list(b))
	j_idx = 2 * match_pixel_num / total_pixel_num
	if j_idx > j_thre:
		return True
	else:
		return False
	
def check_match_overlap(ref_arr, que_arr):
	a = set((tuple(i) for i in ref_arr))
	b = set((tuple(i) for i in que_arr))
	match_pixel_num = len(list(a & b))
	ref_pixel_num = len(list(a))
	if match_pixel_num > (0.5 * ref_pixel_num):
		return True
	else:
		return False
		
def get_f1_score(reference_mask, query_mask, jac):
	reference_mask_coords = get_indices_sparse(reference_mask)[1:]
	query_mask_coords = get_indices_sparse(query_mask)[1:]
	reference_mask_coords = list(map(lambda x: np.array(x).T, reference_mask_coords))
	query_mask_coords = list(map(lambda x: np.array(x).T, query_mask_coords))

	reference_mask_matched_index_list = []
	query_mask_matched_index_list = []


	TP = 0
	for i in range(len(reference_mask_coords)):
		# print(i)
		if len(reference_mask_coords[i]) != 0:
			current_reference_mask_coords = reference_mask_coords[i]
			query_mask_search_num = np.unique(list(map(lambda x: query_mask[tuple(x)], current_reference_mask_coords)))
			# print(query_mask_search_num)
			for j in query_mask_search_num:
				current_query_mask_coords = query_mask_coords[j-1]
				if j != 0 and len(current_query_mask_coords) < 10000:
					if (j-1 not in query_mask_matched_index_list) and (i not in reference_mask_matched_index_list):
						matched_bool = check_match_iou(current_reference_mask_coords, current_query_mask_coords, jac)
						if matched_bool == True:
							TP += 1
							i_ind = i
							j_ind = j - 1
							reference_mask_matched_index_list.append(i_ind)
							query_mask_matched_index_list.append(j_ind)
							break

	reference_mask_cell_num = len(reference_mask_coords)
	query_mask_cell_num = len(query_mask_coords)
	FN_FP = reference_mask_cell_num + query_mask_cell_num - TP * 2
	FN = reference_mask_cell_num - TP
	FP = query_mask_cell_num - TP
	f1_score = TP / (TP + 0.5 * FN_FP)

	return f1_score, TP, FP, FN

def calculate_jaccard(ref_arr, que_arr):
	a = set((tuple(i) for i in ref_arr))
	b = set((tuple(i) for i in que_arr))
	match_pixel_num = len(list(a & b))
	union_pixel_num = len(list(a | b))
	j_idx = match_pixel_num / union_pixel_num
	return j_idx


def get_seg_score(reference_mask, query_mask):
	reference_mask_coords = get_indices_sparse(reference_mask)[1:]
	query_mask_coords = get_indices_sparse(query_mask)[1:]
	reference_mask_coords = list(map(lambda x: np.array(x).T, reference_mask_coords))
	query_mask_coords = list(map(lambda x: np.array(x).T, query_mask_coords))
	
	# reference_mask_matched_index_list = []
	query_mask_matched_index_list = []
	
	seg_score_list = []
	for i in range(len(reference_mask_coords)):
		if len(reference_mask_coords[i]) != 0 and len(reference_mask_coords[i]) < 10000:
			current_reference_mask_coords = reference_mask_coords[i]
			query_mask_search_num = np.unique(list(map(lambda x: query_mask[tuple(x)], current_reference_mask_coords)))
			best_jaccard = 0
			for j in query_mask_search_num:
				current_query_mask_coords = query_mask_coords[j-1]
				if j != 0 and len(current_query_mask_coords) < 10000:
					if (j-1 not in query_mask_matched_index_list):
						match_bool = check_match_overlap(current_reference_mask_coords, current_query_mask_coords)
						if match_bool == True:
							current_jaccard = calculate_jaccard(current_reference_mask_coords, current_query_mask_coords)
							if current_jaccard > best_jaccard:
								best_jaccard = current_jaccard
								# i_ind_best = i
								j_ind_best = j-1
				
			if best_jaccard > 0:
				# reference_mask_matched_index_list.append(i_ind_best)
				query_mask_matched_index_list.append(j_ind_best)
			seg_score_list.append(best_jaccard)

	
	return np.average(seg_score_list)


if __name__ == '__main__':
	data_dir = '/data/HuBMAP/CODEX_annotated/HBM279.TQRS.775'
	tile_list = ['R001_X003_Y004', 'R001_X004_Y003']
	# tile_list = ['R001_X003_Y004']
	expert_list = ['expert1', 'expert2']
	# expert_list = ['expert1']
	method_list = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	           'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new',
	           'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial', 'expert1', 'expert2']
	methods_abre_list = ['DeepCell 0.12.3 cell membrane', 'DeepCell 0.12.3 cytoplasm',
                            'DeepCell 0.9.0 cell membrane',
                            'DeepCell 0.9.0 cytoplasm', 'DeepCell 0.6.0 cell membrane', 'DeepCell 0.6.0 cytoplasm',
                            'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
                            'AICS(classic)',
                            'Cellsegm', 'Voronoi', 'Expert1', 'Expert2']
	# jaccard_list = np.linspace(0, 1, 101)
	pearson_list = []
	# for jaccard in jaccard_list:
	if True:
		seg_score_dataframe = pd.DataFrame(columns=['seg_score', 'quality_score', 'expert', 'Method', 'tile'])
		for tile in tile_list:
			for expert in expert_list:
				for method in method_list:
					mask_expert = pickle.load(bz2.BZ2File(join(data_dir, tile, 'mask_' + expert + '.pickle'), 'rb'))
					mask_method = pickle.load(bz2.BZ2File(join(data_dir, tile, 'mask_' + method + '.pickle'), 'rb'))
					print(method)
					print('cell num', len(np.unique(mask_method)))
					
					f = open(join(data_dir, tile, 'metrics_' + method + '_v1.8.json'))
					metrics = json.load(f)
					current_quality_score = metrics['QualityScore']
					current_seg_score_1 = get_seg_score(mask_expert, mask_method)
					current_seg_score_2 = get_seg_score(mask_method, mask_expert)
					current_seg_score = (current_seg_score_1 + current_seg_score_2) / 2
					if tile == 'R001_X003_Y004':
						tile_name = 'tile1'
					elif tile == 'R001_X004_Y003':
						tile_name = 'tile2'
					method_name = methods_abre_list[method_list.index(method)]
					seg_score_dataframe.loc[len(seg_score_dataframe)] = [current_seg_score, current_quality_score, expert, method_name, tile_name]
					print([current_seg_score, current_quality_score])
		
		seg_score_dataframe.to_csv('/home/hrchen/Documents/Research/hubmap/fig/seg_score_corr_seg.csv')
	
		pearson_list.append(pearsonr(seg_score_dataframe['seg_score'], seg_score_dataframe['quality_score'])[0])
		print(pearson_list)
	seg_score_dataframe = seg_score_dataframe.drop(seg_score_dataframe.loc[seg_score_dataframe['seg_score'] == 1.0].index)

# mask_method = pickle.load(bz2.BZ2File(join(data_dir, tile, 'mask_' + 'cellprofiler' + '.pickle'), 'rb'))
# mask_method = np.load(join(data_dir, tile, 'mask_' + 'cellprofiler' + '.npy'))
#
# plt.imshow(mask_method)
# plt.show()
	seg_score_dataframe = pd.read_csv('/home/hrchen/Documents/Research/hubmap/fig/seg_score_corr_seg.csv', index_col=0)
	seg_score_dataframe = seg_score_dataframe.drop(seg_score_dataframe.loc[seg_score_dataframe['seg_score'] == 1.0].index)

	star = mpath.Path.unit_regular_star(6)

	marker = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o", '<', '>']

	seg_score_dataframe["Annotation"] = seg_score_dataframe["expert"] + ' in ' + seg_score_dataframe["tile"]
	r = pearsonr(seg_score_dataframe.values[:,0], seg_score_dataframe.values[:,1])[0]

	r = "%.2f" % r
	
	
	fig, ax = plt.subplots()
	p = sns.scatterplot(data=seg_score_dataframe, x='seg_score', y='quality_score', hue='Annotation', style='Method', markers=marker, s=60, lw=0)
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
	# ax.set(ylabel=None)
	# ax.set_yticklabels([])
	# ax.legend(bbox_to_anchor=(0.5, -0.15),ncol=2)
	plt.xlabel('SEG\' Score')
	plt.ylabel('Quality Score')
	
	plt.savefig('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/seg_score_corr_prime.png', bbox_inches='tight',dpi=500)
	# plt.show()
	plt.clf()
	
	seg_score_expert1 = seg_score_dataframe[seg_score_dataframe['expert'] == 'expert1']
	pearsonr(seg_score_expert1['seg_score'], seg_score_expert1['quality_score'])

	seg_score_expert2 = seg_score_dataframe[seg_score_dataframe['expert'] == 'expert2']
	pearsonr(seg_score_expert2['seg_score'], seg_score_expert2['quality_score'])
	
	
	
	
	f1_score_dataframe = pd.read_csv('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/f1_score/f1_score_corr_expert_0.3.csv', index_col=0)
	# f1_score_dataframe['F1_score'] = f1_score_dataframe['TP'] / (f1_score_dataframe['TP'] + 0.5 * (f1_score_dataframe['FP'] + f1_score_dataframe['FN']))
	
	r = pearsonr(f1_score_dataframe.values[:,0], f1_score_dataframe.values[:,1])[0]
	r = "%.2f" % r
	
	
	marker = [star, "*", "X", "P", "^", "v", "H", "h", "p", "o", 'D', "s", "d", "o", '<', '>']
	
	f1_score_dataframe["Annotation"] = f1_score_dataframe["expert"] + ' in ' + f1_score_dataframe["tile"]
	
	fig, ax = plt.subplots()
	p = sns.scatterplot(data=f1_score_dataframe, x='f1_score', y='quality_score', hue='Annotation', style='Method',
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

	
	# ax.legend(bbox_to_anchor=(0.5, -0.15),ncol=2)
	plt.xlabel('F1 Score')
	plt.ylabel('Quality Score')
	# plt.yticks()
	plt.savefig('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/F1_corr_test.png', bbox_inches='tight',
	            dpi=500)
	# plt.show()
	plt.clf()





	jaccard_list = np.linspace(0, 1, 101)
	pearson_list = []
	for jaccard in jaccard_list:
	# if True:
		f1_score_dataframe = pd.DataFrame(columns=['f1_score', 'quality_score', 'expert', 'Method', 'tile', 'jaccard_thre', 'TP', 'FP', 'FN'])
		for tile in tile_list:
			for expert in expert_list:
				for method in method_list:
					if method != expert:
						mask_expert = pickle.load(bz2.BZ2File(join(data_dir, tile, 'mask_' + expert + '.pickle'), 'rb'))
						mask_method = pickle.load(bz2.BZ2File(join(data_dir, tile, 'mask_' + method + '.pickle'), 'rb'))
						print(method)
						print('cell num', len(np.unique(mask_method)))
						
						f = open(join(data_dir, tile, 'metrics_' + method + '_v1.8.json'))
						metrics = json.load(f)
						current_quality_score = metrics['QualityScore']
						current_f1_score, TP, FP, FN = get_f1_score(mask_expert, mask_method, jaccard)
						if tile == 'R001_X003_Y004':
							tile_name = 'tile1'
						elif tile == 'R001_X004_Y003':
							tile_name = 'tile2'
						method_name = methods_abre_list[method_list.index(method)]
						f1_score_dataframe.loc[len(f1_score_dataframe)] = [current_f1_score, current_quality_score, expert,
						                                                     method_name, tile_name, jaccard, TP, FP, FN]
						print([current_f1_score, current_quality_score, TP, FP, FN])
			
		f1_score_dataframe.to_csv('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/f1_score/f1_score_corr_expert_' + str(jaccard) + '.csv')
	
	
	
	
	
	f1_score_dir_list = sorted(glob.glob('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/f1_score/f1_score_corr_*.csv'))
	for f1_score_dir in f1_score_dir_list:
		f1_score_current = pd.read_csv(f1_score_dir, index_col=0)
		print(f1_score_current['jaccard_thre'])
		if f1_score_dir == f1_score_dir_list[0]:
			f1_score_dataframe = f1_score_current
		else:
			f1_score_dataframe = pd.concat([f1_score_dataframe, f1_score_current], axis=0)

	
	PRAUC_score_dataframe = pd.DataFrame(columns=['PRAUC', 'quality_score', 'expert', 'Method', 'tile'])
	for tile in ['tile1', 'tile2']:
		for expert in expert_list:
			for method in methods_abre_list:
				if expert != 'e' + method[1:]:
					current_f1_score = f1_score_dataframe.loc[(f1_score_dataframe['Method'] == method) & (f1_score_dataframe['expert'] == expert) & (f1_score_dataframe['tile'] == tile)]
					current_precision = current_f1_score['TP'] / (current_f1_score['TP'] + current_f1_score['FP'])
					current_recall = current_f1_score['TP'] / (current_f1_score['TP'] + current_f1_score['FN'])
					pr_matrix = np.stack((current_precision, current_recall), axis=0).T
					pr_matrix_sorted = pr_matrix[(-pr_matrix[:, 0]).argsort()]
					current_PRAUC = auc(pr_matrix_sorted[:, 0], pr_matrix_sorted[:, 1])
					quality_score = np.unique(current_f1_score['quality_score'])[0]
					PRAUC_score_dataframe.loc[len(PRAUC_score_dataframe)] = [current_PRAUC, quality_score, expert, method, tile]
			
	
	r = pearsonr(PRAUC_score_dataframe.values[:,0], PRAUC_score_dataframe.values[:,1])[0]
	r = "%.2f" % r
	marker = [star, "*", "X", "P", "^", "v", "H", "h", "p", "o", 'D', "s", "d", "o", '<', '>']
	
	PRAUC_score_dataframe["Annotation"] = PRAUC_score_dataframe["expert"] + ' in ' + PRAUC_score_dataframe["tile"]
	
	fig, ax = plt.subplots()
	p = sns.scatterplot(data=PRAUC_score_dataframe, x='PRAUC', y='quality_score', hue='Annotation', style='Method',
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
	# ax.set(ylabel=None)
	# ax.set_yticklabels([])
	ax.get_legend().remove()
	# ax.legend(bbox_to_anchor=(0.5, -0.15),ncol=2)
	plt.xlabel('PRAUC')
	plt.ylabel('Quality Score')
	# plt.yticks()
	plt.savefig('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/PRAUC_corr_test.png', bbox_inches='tight', dpi=500)
	# plt.show()
	plt.clf()

	PRAUC_score_expert1 = PRAUC_score_dataframe[PRAUC_score_dataframe['expert'] == 'expert1']
	pearsonr(PRAUC_score_expert1['PRAUC'], PRAUC_score_expert1['quality_score'])
	
	PRAUC_score_expert2 = PRAUC_score_dataframe[PRAUC_score_dataframe['expert'] == 'expert2']
	pearsonr(PRAUC_score_expert2['PRAUC'], PRAUC_score_expert2['quality_score'])






	'''
	AUC_score_dataframe = pd.DataFrame(columns=['AUC', 'quality_score', 'expert', 'Method', 'tile'])
	for tile in ['tile1', 'tile2']:
		for expert in expert_list:
			for method in methods_abre_list:
				if expert != 'e' + method[1:]:
					current_f1_score = f1_score_dataframe.loc[
						(f1_score_dataframe['Method'] == method) & (f1_score_dataframe['expert'] == expert) & (
									f1_score_dataframe['tile'] == tile)]
					current_precision = current_f1_score['TP'] / (current_f1_score['TP'] + current_f1_score['FP'])
					current_recall = current_f1_score['TP'] / (current_f1_score['TP'] + current_f1_score['FN'])
					pr_matrix = np.stack((current_precision, current_recall), axis=0).T
					pr_matrix_sorted = pr_matrix[(-pr_matrix[:, 0]).argsort()]
					current_AUC = auc(pr_matrix_sorted[:, 0], pr_matrix_sorted[:, 1])
					quality_score = np.unique(current_f1_score['quality_score'])[0]
					AUC_score_dataframe.loc[len(AUC_score_dataframe)] = [current_AUC, quality_score, expert, method,
					                                                         tile]

	
	r = pearsonr(AUC_score_dataframe.values[:, 0], AUC_score_dataframe.values[:, 1])[0]
	r = "%.2f" % r
	marker = [star, "*", "X", "P", "^", "v", "H", "h", "p", "o", 'D', "s", "d", "o", '<', '>']
	
	AUC_score_dataframe["Annotation"] = AUC_score_dataframe["expert"] + ' in ' + AUC_score_dataframe["tile"]
	
	fig, ax = plt.subplots()
	p = sns.scatterplot(data=AUC_score_dataframe, x='AUC', y='quality_score', hue='Annotation', style='Method',
	                    markers=marker, s=50)
	handles, labels = ax.get_legend_handles_labels()
	# labels = ['compared with expert1 in tile1', 'compared with expert2 in tile1', 'compared with expert1 in tile2', 'compared with expert2 in tile2', 'r=' + r]
	# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
	# handles.append(extra)
	# ax.legend(handles=handles, labels=labels, loc=4)
	h, l = ax.get_legend_handles_labels()
	l1 = ax.legend(h[:5], l[:5], loc='lower right')
	l2 = ax.legend(h[5:], l[5:], loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first
	# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
	ax.set(ylabel=None)
	ax.set_yticklabels([])
	
	# ax.legend(bbox_to_anchor=(0.5, -0.15),ncol=2)
	plt.xlabel('AUC')
	# plt.ylabel('Quality Score')
	# plt.yticks()
	plt.savefig('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/AUC_corr_test.png', bbox_inches='tight',
	            dpi=500)
	# plt.show()
	plt.clf()
	
	AUC_score_expert1 = AUC_score_dataframe[AUC_score_dataframe['expert'] == 'expert1']
	pearsonr(AUC_score_expert1['AUC'], AUC_score_expert1['quality_score'])
	
	AUC_score_expert2 = AUC_score_dataframe[AUC_score_dataframe['expert'] == 'expert2']
	pearsonr(AUC_score_expert2['AUC'], AUC_score_expert2['quality_score'])'''