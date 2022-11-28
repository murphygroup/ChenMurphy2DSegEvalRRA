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
	# parser = argparse.ArgumentParser(description='Run segmentation and evaluation pipeline.')
	# parser.add_argument('script_dir')
	# parser.add_argument('data_dir')
	# args = parser.parse_args()
	script_dir = '/home/hrchen/Documents/Research/hubmap/script'
	save_dir = '/home/hrchen/Documents/Research/hubmap/fig/manuscript v21 fig v2'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	data_types = ['CODEX', 'CellDIVE', 'IMC', 'MIBI']
	# data_type = 'CellDIVE'
	metric_matrix_stack_pieces = []
	for data_type in data_types:
		data_dir = '/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/gaussian/repaired/' + data_type + '/All/concatenated_compartments'
		feature_matrix = np.load(join(data_dir, 'feature_3D.npy'))
		feature_matrix = feature_matrix[:int(feature_matrix.shape[0]/4), :, :]
		metric_matrix_temp_stack = np.zeros((feature_matrix.shape[0], feature_matrix.shape[2], feature_matrix.shape[1], feature_matrix.shape[1]))
		for i in range(feature_matrix.shape[1]-1):
			for j in range(i+1, feature_matrix.shape[1]):
				metric_matrix_temp_stack[:, :, i, j] = abs(feature_matrix[:, i, :] - feature_matrix[:, j, :])
				metric_matrix_temp_stack[:, :, j, i] = abs(feature_matrix[:, i, :] - feature_matrix[:, j, :])

		# mask_list = ['deepmem_new', 'deepmem', 'deepcell_cyto_new', 'deepcell_cyto', 'cellpose_new', 'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
		# # mask_list = ['deepmem', 'deepcell_cyto', 'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
		# method_num = len(mask_list)
		# metric_matrix = np.empty((10, method_num, method_num))
		# index = 0
		# img_list = sorted(glob.glob(data_dir + '/**/result_pairwise', recursive=True))
		# img = img_list[0]
		# metric_matrix_temp_stack_pieces = []
		# for img in img_list:
		# 	# img_dir = join(checklist[i, 0], checklist[i, 2])
		# 	for i in range(method_num-1):
		# 		for j in range(i+1, method_num):
		# 			try:
		# 				metrics = np.loadtxt(join(img, mask_list[i] + '_' + mask_list[j] + '_pairwise.txt'))
		# 			except:
		# 				metrics = np.zeros((10)).tolist()
		# 			metric_matrix[:, i, j] = metrics
		# 			# for method in me
		#
		# 			# if os.path.exists(join(random_dir, 'pairwise_metric_matrix_7.npy')):
		# 			# 	metric_matrix_7 = np.load(join(random_dir, 'pairwise_metric_matrix_7.npy'))
		# 			# 	metric_matrix_7[:, :6, :6] = metric_matrix[:, :6, :6]
		# 			# 	metric_matrix = metric_matrix_7
		# 			# 	del metric_matrix_7
		# 			# try:
		# 			# 	cell_num_cellsegm = np.loadtxt(join(random_dir, 'result', 'cell_basic_cellsegm.txt'))[0]
		# 			# 	if cell_num_cellsegm < 20:
		# 			# 		metric_matrix[:, 1, :] = np.nan
		# 			# 		metric_matrix[:, :, 1] = np.nan
		# 			# except:
		# 			# 	pass
		# 			# metric_matrix = np.delete(metric_matrix, 4, 0)
		# 			# metric_matrix = np.delete(metric_matrix, 28, 0)
		# 			# method_num = metric_matrix.shape[2]
		# 	metric_matrix = np.nan_to_num(metric_matrix, 0)
		# 	metric_matrix_temp = np.expand_dims(metric_matrix.copy(), 0)
		# 	metric_matrix_temp_stack_pieces.append(metric_matrix_temp)
		# 	# if index == 0:
		# 	# 	metric_matrix_temp_stack = metric_matrix_temp
		# 	# else:
		# 	# 	metric_matrix_temp_stack = np.vstack((metric_matrix_temp_stack, metric_matrix_temp))
		# 	index += 1
		# metric_matrix_temp_stack = np.vstack(metric_matrix_temp_stack_pieces)
		#
		# # 0 means cloest
		#
		# flip_list = [0, 1, 2, 3, 6, 8, 9]
		# # normalize_list = [0, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 39, 40, 41, 42, 43, 44, 45, 46, 47]
		# # normalize_list = [0, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 40, 41, 42, 43]
		# normalize_list = [0, 4, 5, 6, 7]
		# for p in range(metric_matrix_temp_stack.shape[0]):
		# 	for c in range(metric_matrix_temp_stack.shape[1]):
		# 		metric_matrix_temp_stack[p, c, ...] = get_symmetric_metrics(metric_matrix_temp_stack[p, c, ...])
		# 		if c in normalize_list:
		# 			metric_matrix_temp_stack[p, c, ...] = get_normalized_metrics(metric_matrix_temp_stack[p, c, ...])
		# 		# [-num, -ji, -dc, -tauc, hd, bce, -mi, voi, -kap, -vs]
		# 		if c in flip_list:
		# 			metric_matrix_temp_stack[p, c, ...] = get_flipped_metrics(metric_matrix_temp_stack[p, c, ...])
		#
		metric_matrix_temp_stack[np.where(metric_matrix_temp_stack == 0)] = 1
		for i in range(metric_matrix_temp_stack.shape[0]):
			for j in range(metric_matrix_temp_stack.shape[1]):
				matrix = metric_matrix_temp_stack[i, j, :, :]
				matrix[np.diag_indices_from(matrix)] = 0
				metric_matrix_temp_stack[i, j, :, :] = matrix
		
		metric_matrix_stack_pieces.append(metric_matrix_temp_stack)
	metric_matrix_stack = np.vstack(metric_matrix_stack_pieces)
	# 	if index == 0:
	# 		metric_matrix_final = metric_matrix
	# 	else:
	# 		metric_matrix_final = metric_matrix_final + metric_matrix
	# 		# print(np.sum(metric_matrix_final))
	# metric_matrix_final = metric_matrix_final / index
	# sns.set(rc={'figure.figsize':(10,8)})
	methods_abre = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto',
	                'DeepCell 0.9.0 mem',
	                'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
	                'AICS(classic)',
	                'Cellsegm', 'Voronoi']
	
	methods_abre_ordered = ['Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1',
	                        'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                        'DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto',
	                        'DeepCell 0.9.0 mem', 'DeepCell 0.9.0 cyto',
	                        'CellX','AICS(classic)','CellProfiler','Voronoi','Cellsegm'
	                        ]
	
	
	# methods_abre_ordered = ['Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto', 'DeepCell 0.9.0 mem', 'DeepCell 0.9.0 cyto', 'CellProfiler', 'AICS(classic)', 'CellX',  'Voronoi', 'Cellsegm']

	# methods_abre = ['deepmem', 'deepcell_cyto', 'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
	# methods_abre_ordered = ['deepmem', 'deepcell_cyto', 'cellpose', 'aics_classic', 'cellprofiler', 'CellX', 'artificial', 'cellsegm']
	# methods_abre = [	'cellprofiler',
	#                     'cellsegm',
	#                     'cellX',
	#                     'deepcell_c',
	#                     'cellpose',
	#                     'deepcell_m',
	#                     'aics_cls']
	# methods_abre_ordered = ['cellpose',	'cellprofiler', 'deepcell_m', 'cellX', 'aics_cls', 'deepcell_c', 'cellsegm']

	# ax = sns.heatmap(np.average(np.average(metric_matrix_temp_stack, axis=0), axis=0))
	# metric_matrix_stack = np.average(metric_matrix_stack, axis=0)
	# metric_matrix_stack_2d = metric_matrix_stack.reshape(metric_matrix_stack.shape[0],
	#                                                      metric_matrix_stack.shape[1] * metric_matrix_stack.shape[2]).T
	metric_matrix_stack_2d = metric_matrix_stack.reshape(metric_matrix_stack.shape[0]*metric_matrix_stack.shape[1], metric_matrix_stack.shape[2]* metric_matrix_stack.shape[3]).T

	ss = StandardScaler().fit(metric_matrix_stack_2d)
	metric_matrix_stack_2d_scaled = ss.transform(metric_matrix_stack_2d)
	pca = PCA(n_components=1).fit(metric_matrix_stack_2d_scaled)
	metric_matrix_stack_2d_pca = pca.transform(metric_matrix_stack_2d_scaled)
	method_num = 14
	avg_metric_matrix = metric_matrix_stack_2d_pca.reshape(method_num, method_num)
	avg_metric_matrix = avg_metric_matrix - avg_metric_matrix[1, 1]
	# avg_metric_matrix = np.average(np.average(metric_matrix_temp_stack, axis=0), axis=0)
	# avg_metric_matrix = np.average(np.average(metric_matrix_temp_stack, axis=0), axis=0)
	# avg_metric_matrix[np.diag_indices_from(avg_metric_matrix)] = 0
	avg_metric_matrix_df = pd.DataFrame(avg_metric_matrix, index=methods_abre, columns=methods_abre)
	# avg_metric_matrix_df = avg_metric_matrix_df.sort_values(by=['cellpose'], axis=1)
	# avg_metric_matrix_df = avg_metric_matrix_df.sort_values(by=['cellpose'], axis=0)
	avg_metric_matrix_df = avg_metric_matrix_df.reindex(methods_abre_ordered, axis=1)
	avg_metric_matrix_df = avg_metric_matrix_df.reindex(methods_abre_ordered, axis=0)
	# avg_metric_matrix_df.values[-1, :] = avg_metric_matrix_df.values[-1, :]-50
	# avg_metric_matrix_df.values[:, -1] = avg_metric_matrix_df.values[:, -1]-50
	# avg_metric_matrix_df.values[-1, -1] = 0
	
	
	ax = sns.heatmap(avg_metric_matrix_df, cmap='magma')
	ax.tick_params(axis='both', which='major', labelsize=13)
	cax = ax.figure.axes[-1]
	cax.tick_params(labelsize=13)
	
	plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
	plt.yticks(np.arange(5)+0.5)
	plt.yticks(np.arange(0.5, 14.5, 1))
	plt.ylim(0, 14)
	plt.tight_layout()
	plt.savefig(join(save_dir, 'pairwise_metrics_single.png'), dpi=500)
	# np.savetxt(join(save_dir, 'pairwise_metrics.txt'), avg_metric_matrix)
	avg_metric_matrix_df.to_csv(join(save_dir, 'pairwise_metrics.csv'))
	plt.clf()
	plt.close()
	# 1579 1828
	#
	#
	#
	# deepcell_cyto 87 deepmem 624 aics 982
	# cellprofiler 396 cellsegm 1 cellx 1025
	#
	#
	# cellprofiler
	# cellsegm
	# cellX
	# deepcell_cyto
	# cellpose
	# deepmem
	# aics_cls

	# cellpose 526
	# CellX 1020
	# deepcell cyto 45
	# deepcell_mem 548
