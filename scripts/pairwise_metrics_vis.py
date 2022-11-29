import numpy as np
import os
from os.path import join
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import bz2
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_symmetric_metrics(matrix):
	for i in range(method_num-1):
		for j in range((i+1), method_num):
			matrix[j, i] = matrix[i, j]
	for i in range(method_num):
		matrix[i, i] = 0
	return matrix

def get_normalized_metrics(matrix):
	# metric_sum = np.sum(matrix) / 2
	metric_max = np.max(matrix)
	if metric_max != 0:
		# metric_min = np.min(matrix[:method_num-1, 1:])
		for i in range(method_num):
			for j in range(method_num):
				if i != j:
					# matrix[i, j] = (matrix[i, j] - metric_min) / (metric_max - metric_min)
					matrix[i, j] = matrix[i, j] / metric_max
	return matrix

def get_flipped_metrics(matrix):
	matrix = 1 - matrix
	# matrix[np.where(matrix == 1)] = 0
	return matrix

if __name__ == '__main__':

	
	
	file_dir = os.getcwd()
	save_dir = join(file_dir, 'figures')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	data_types = ['CODEX', 'CellDIVE', 'IMC', 'MIBI']
	# data_type = 'CellDIVE'
	metric_matrix_stack_pieces = []
	for data_type in data_types:
		data_dir = join(file_dir, 'data', 'intermediate', 'manuscript_v28_repaired_pairwise', data_type)

		mask_list = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
		           'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0',
		           'cellpose_new',
		           'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
		# mask_list = ['deepcell_membrane', 'deepcell_cytoplasm', 'deepcell_membrane_new', 'deepcell_cytoplasm_new', 'cellpose', 'cellpose_new', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
		method_num = len(mask_list)
		metric_matrix = np.empty((10, method_num, method_num))
		index = 0
		img_list = sorted(glob.glob(data_dir + '/**/random_gaussian_0/result_pairwise_repaired', recursive=True))
		img = img_list[0]
		metric_matrix_temp_stack_pieces = []
		for img in img_list:
			# img_dir = join(checklist[i, 0], checklist[i, 2])
			for i in range(method_num-1):
				for j in range(i+1, method_num):
					if os.path.exists(join(img, mask_list[i] + '_' + mask_list[j] + '_pairwise.txt')):
						metrics = np.loadtxt(join(img, mask_list[i] + '_' + mask_list[j] + '_pairwise.txt'))

					elif os.path.exists(join(img, mask_list[j] + '_' + mask_list[i] + '_pairwise.txt')):
						metrics = np.loadtxt(join(img, mask_list[j] + '_' + mask_list[i] + '_pairwise.txt'))
					else:
						metrics = np.zeros((10)).tolist()
					if len(metrics) == 0:
						metrics = np.zeros((10)).tolist()
					metric_matrix[:, i, j] = metrics

			metric_matrix = np.nan_to_num(metric_matrix, 0)
			metric_matrix_temp = np.expand_dims(metric_matrix.copy(), 0)
			metric_matrix_temp_stack_pieces.append(metric_matrix_temp)

			index += 1
		metric_matrix_temp_stack = np.vstack(metric_matrix_temp_stack_pieces)
	
		# 0 means cloest
		
		flip_list = [0, 1, 2, 3, 6, 8, 9]
		normalize_list = [0, 4, 5, 6, 7]
		for p in range(metric_matrix_temp_stack.shape[0]):
			for c in range(metric_matrix_temp_stack.shape[1]):
				metric_matrix_temp_stack[p, c, ...] = get_symmetric_metrics(metric_matrix_temp_stack[p, c, ...])
				if c in normalize_list:
					metric_matrix_temp_stack[p, c, ...] = get_normalized_metrics(metric_matrix_temp_stack[p, c, ...])
				# [-num, -ji, -dc, -tauc, hd, bce, -mi, voi, -kap, -vs]
				if c in flip_list:
					metric_matrix_temp_stack[p, c, ...] = get_flipped_metrics(metric_matrix_temp_stack[p, c, ...])
		
		metric_matrix_temp_stack[np.where(metric_matrix_temp_stack == 0)] = 1
		for i in range(metric_matrix_temp_stack.shape[0]):
			for j in range(metric_matrix_temp_stack.shape[1]):
				matrix = metric_matrix_temp_stack[i, j, :, :]
				matrix[np.diag_indices_from(matrix)] = 0
				metric_matrix_temp_stack[i, j, :, :] = matrix
		
		metric_matrix_stack_pieces.append(metric_matrix_temp_stack)
	metric_matrix_stack = np.vstack(metric_matrix_stack_pieces)

	methods_abre = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto',
	                            'DeepCell 0.9.0 mem',
	                            'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                            'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
	                            'AICS(classic)',
	                            'Cellsegm', 'Voronoi']
	# this order is determined by the linkage from clustering in seaborn.clustermap, manual setting for better visualization
	methods_abre_ordered = ['Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto',\
	'DeepCell 0.9.0 mem', 'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',\
	'CellProfiler', 'AICS(classic)', 'CellX', 'Voronoi','Cellsegm']
	metric_matrix_stack_2d = metric_matrix_stack.reshape(metric_matrix_stack.shape[0]*metric_matrix_stack.shape[1], metric_matrix_stack.shape[2]* metric_matrix_stack.shape[3]).T
	
	ss = StandardScaler().fit(metric_matrix_stack_2d)
	metric_matrix_stack_2d_scaled = ss.transform(metric_matrix_stack_2d)
	pca = PCA(n_components=1).fit(metric_matrix_stack_2d_scaled)
	metric_matrix_stack_2d_pca = pca.transform(metric_matrix_stack_2d_scaled)
	avg_metric_matrix = metric_matrix_stack_2d_pca.reshape(method_num, method_num)
	avg_metric_matrix = avg_metric_matrix[1, 1] - avg_metric_matrix
	# avg_metric_matrix = np.average(np.average(metric_matrix_temp_stack, axis=0), axis=0)
	# avg_metric_matrix = np.average(np.average(metric_matrix_temp_stack, axis=0), axis=0)
	# avg_metric_matrix[np.diag_indices_from(avg_metric_matrix)] = 0
	avg_metric_matrix_df = pd.DataFrame(avg_metric_matrix, index=methods_abre, columns=methods_abre)
	# avg_metric_matrix_df = avg_metric_matrix_df.sort_values(by=['cellpose'], axis=1)
	# avg_metric_matrix_df = avg_metric_matrix_df.sort_values(by=['cellpose'], axis=0)
	avg_metric_matrix_df = avg_metric_matrix_df.reindex(methods_abre_ordered, axis=1)
	avg_metric_matrix_df = avg_metric_matrix_df.reindex(methods_abre_ordered, axis=0)
	

	
	ax = sns.heatmap(avg_metric_matrix_df,  cmap='magma', vmax=250)
	plt.xticks(rotation=50, ha='right', rotation_mode='anchor', fontsize=12)
	cbar = ax.collections[0].colorbar
	cbar.ax.tick_params(labelsize=12)
	
	plt.yticks(np.arange(0.5, 14.5, 1), fontsize=12)
	plt.ylim(0, 14)
	plt.tight_layout()
	plt.savefig(join(save_dir, 'pairwise_metrics.png'), dpi=400)
	np.savetxt(join(save_dir, 'pairwise_metrics.txt'), avg_metric_matrix)
	avg_metric_matrix_df.to_csv(join(save_dir, 'pairwise_metrics.csv'))
	plt.clf()
	plt.close()

