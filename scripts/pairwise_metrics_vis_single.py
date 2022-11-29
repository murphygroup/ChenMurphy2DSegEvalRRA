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
		data_dir = join(file_dir, 'data', 'metrics', 'gaussian', 'repaired', data_type, 'all_tissues/concatenated_compartments')
		feature_matrix = pickle.load(bz2.BZ2File(join(data_dir, 'feature_3D.pickle'), 'rb'))
		feature_matrix = feature_matrix[:int(feature_matrix.shape[0]/4), :, :]
		metric_matrix_temp_stack = np.zeros((feature_matrix.shape[0], feature_matrix.shape[2], feature_matrix.shape[1], feature_matrix.shape[1]))
		for i in range(feature_matrix.shape[1]-1):
			for j in range(i+1, feature_matrix.shape[1]):
				metric_matrix_temp_stack[:, :, i, j] = abs(feature_matrix[:, i, :] - feature_matrix[:, j, :])
				metric_matrix_temp_stack[:, :, j, i] = abs(feature_matrix[:, i, :] - feature_matrix[:, j, :])


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
	methods_abre_ordered = ['Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1',
	                        'DeepCell 0.9.0 mem', 'DeepCell 0.9.0 cyto',
	                        'DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto',
	                        
	                        'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                        'CellProfiler','AICS(classic)','CellX','Voronoi','Cellsegm'
	                        ]
	
	

	metric_matrix_stack_2d = metric_matrix_stack.reshape(metric_matrix_stack.shape[0]*metric_matrix_stack.shape[1], metric_matrix_stack.shape[2]* metric_matrix_stack.shape[3]).T

	ss = StandardScaler().fit(metric_matrix_stack_2d)
	metric_matrix_stack_2d_scaled = ss.transform(metric_matrix_stack_2d)
	pca = PCA(n_components=1).fit(metric_matrix_stack_2d_scaled)
	metric_matrix_stack_2d_pca = pca.transform(metric_matrix_stack_2d_scaled)
	method_num = 14
	avg_metric_matrix = metric_matrix_stack_2d_pca.reshape(method_num, method_num)
	avg_metric_matrix = avg_metric_matrix - avg_metric_matrix[1, 1]

	avg_metric_matrix_df = pd.DataFrame(avg_metric_matrix, index=methods_abre, columns=methods_abre)

	avg_metric_matrix_df = avg_metric_matrix_df.reindex(methods_abre_ordered, axis=1)
	avg_metric_matrix_df = avg_metric_matrix_df.reindex(methods_abre_ordered, axis=0)

	
	
	ax = sns.heatmap(avg_metric_matrix_df, cmap='magma', vmax=250)
	
	plt.xticks(rotation=50, ha='right', rotation_mode='anchor', fontsize=12)
	cbar = ax.collections[0].colorbar
	cbar.ax.tick_params(labelsize=12)
	
	plt.yticks(np.arange(0.5, 14.5, 1), fontsize=12)
	plt.ylim(0, 14)
	plt.tight_layout()
	plt.savefig(join(save_dir, 'pairwise_metrics_single.png'), dpi=400)
	avg_metric_matrix_df.to_csv(join(save_dir, 'pairwise_metrics.csv'))
	plt.clf()
	plt.close()

