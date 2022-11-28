import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from os.path import join
import pandas as pd
from sklearn.preprocessing import normalize, scale, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from matplotlib.markers import MarkerStyle
import pandas as pd
import pickle
from matplotlib import gridspec


def visualization_avg(feature_matrix, loading):
	# print(feature_matrix)
	
	if neg_flag_PC1 == True:
		avg_pc1 = -np.average(feature_matrix[:, 0])
	else:
		avg_pc1 = np.average(feature_matrix[:, 0])
	
	if neg_flag_PC2 == True:
		avg_pc2 = -np.average(feature_matrix[:, 1])
	else:
		avg_pc2 = np.average(feature_matrix[:, 1])
	
	if method_type == 'less':
		plt.scatter(avg_pc1, avg_pc2, color=cmap[i],
		            s=50, marker=marker[i], label=methods_abre[i])
	if method_type == 'all':
		if i not in less_methods:
			
			plt.scatter(avg_pc1, avg_pc2, color=cmap[i],
			            s=50, marker=marker[i], label=methods_abre[i])
		else:
			plt.scatter(avg_pc1, avg_pc2, color=cmap[i],
			            s=50, marker=marker[i], label='_no_legend_')
	
	return avg_pc1 * loading[0] + avg_pc2 * loading[1]


# if neg_flag == True:
# 	plt.errorbar(np.average(-feature_matrix[:, 0]), np.average(feature_matrix[:, 1]), xerr=np.std(-feature_matrix[:, 0]), linestyle='None', color=cmap[i])
# else:
# 	plt.errorbar(np.average(feature_matrix[:, 0]), np.average(feature_matrix[:, 1]), xerr=np.std(feature_matrix[:, 0]), linestyle='None', color=cmap[i])
# print(np.average(feature_matrix[:, 0]))
def visualization_avg_fill(feature_matrix):
	if modality == 'CODEX':
		plt.scatter(np.average(feature_matrix[:, 0]), np.average(feature_matrix[:, 1]), color=cmap[i],
		            s=10, marker=MarkerStyle(marker[i], fillstyle=fill[index]), label=methods_abre[i])
		plt.scatter(np.average(feature_matrix[:, 0]), np.average(feature_matrix[:, 1]), color=cmap[i],
		            s=52, marker=marker[i], edgecolors=cmap[i], facecolors='none')
	else:
		plt.scatter(np.average(feature_matrix[:, 0]), np.average(feature_matrix[:, 1]), color=cmap[i],
		            s=10, marker=MarkerStyle(marker[i], fillstyle=fill[index]), label='_nolabel_')
		plt.scatter(np.average(feature_matrix[:, 0]), np.average(feature_matrix[:, 1]), color=cmap[i],
		            s=52, marker=marker[i], edgecolors=cmap[i], facecolors='none')


def visualization_track(track, method_num):
	if neg_flag_PC1 == True:
		track[:, 0] = -track[:, 0]
	
	if neg_flag_PC2 == True:
		track[:, 1] = -track[:, 1]
	
	plt.plot(track[:, 0], track[:, 1], color=cmap[method_num])
	plt.scatter(track[:, 0], track[:, 1], edgecolors=cmap[method_num], s=10, facecolors='none')


def visualization_new_method(new_method_num):
	# point = np.load(join(data_dir, modality, 'All', compartment, 'feature_3D_' + new_methods[new_method_num] + '.npy'))
	point = features_3D_new_methods[:, new_method_num, :]
	data_length = int(features_3D_new_methods.shape[0] / (noise_num + 1))
	point = point[:data_length, :]
	point = PCA_model.transform(ss_model.transform(point.copy()))
	# print(new_method_num)
	# print(new_methods_abre)
	# print(marker)
	if neg_flag == True:
		
		plt.scatter(-np.average(point[:, 0]), np.average(point[:, 1]), color=cmap[new_method_num + 7],
		            s=10, marker=marker[new_method_num + 7], label=new_methods_abre[new_method_num])
	else:
		plt.scatter(np.average(point[:, 0]), np.average(point[:, 1]), color=cmap[new_method_num + 7],
		            s=10, marker=marker[new_method_num + 7], label=new_methods_abre[new_method_num])
	if neg_flag == True:
		# lower_error = abs(min(-point[:, 0]) - np.average(-point[:, 0]))
		# upper_error = abs(min(-point[:, 0]) - np.average(-point[:, 0]))
		# plt.errorbar(np.average(-point[:, 0]), np.average(point[:, 1]), xerr=[np.array([lower_error]), np.array([upper_error])], linestyle='None', color=cmap[new_method_num+7])
		plt.errorbar(np.average(-point[:, 0]), np.average(point[:, 1]), xerr=np.std(-point[:, 0]), linestyle='None',
		             color=cmap[new_method_num + 7])
	else:
		# lower_error = abs(min(point[:, 0]) - np.average(point[:, 0]))
		# upper_error = abs(min(point[:, 0]) - np.average(point[:, 0]))
		# plt.errorbar(np.average(point[:, 0]), np.average(point[:, 1]), xerr=[np.array([lower_error]), np.array([upper_error])], linestyle='None', color=cmap[new_method_num+7])
		plt.errorbar(np.average(point[:, 0]), np.average(point[:, 1]), xerr=np.std(point[:, 0]), linestyle='None',
		             color=cmap[new_method_num + 7])


def get_stack_features(features):
	feature_2d = np.empty((0, features.shape[2]))
	for i in range(method_num):
		feature_method = features[:, i, :]
		feature_2d = np.vstack((feature_2d, feature_method))
	return feature_2d


def get_PCA_scaled_score(features):
	score_max = features[-2, 0]
	score_min = features[-1, 0]
	# print((score_max, score_min))
	features = -(features - score_min) / (score_max - score_min)
	return features[:-2]


def get_sigmoid(feature_matrix):
	feature_matrix[:, -6:] = 1 / (1 + np.exp(-feature_matrix.copy()[:, -6:]))
	return feature_matrix


def get_PCA_features(features):
	print(features.shape)
	# for k in range(features.shape[0]):
	# 	print(features[k])
	# 	if np.sum(features[k]) == 0:
	# 		print(k)
	save_dir = join('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8', 'merge', 'all_modalities',
	                compartment, modality_type, tissue)
	if modality_type == 'all' and len(noise_type) > 1:
		ss = StandardScaler().fit(features)
		
		feature_2d_scaled = ss.transform(features)
		
		pca = PCA(n_components=2).fit(feature_2d_scaled)
		for n_com in range(2):
			if np.sum(pca.components_.T[:, n_com]) < 0:
				pca.components_.T[:, n_com] = -pca.components_.T[:, n_com]
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		pickle.dump([ss, pca], open(join(save_dir, 'pca_10_metrics.pickle'), "wb"))
	print(pca.explained_variance_ratio_)



def visualization_loadings(data):
	PC = ['PC1', 'PC2']
	for i in range(2):
		sns.barplot(x=np.arange(1, data.shape[0] + 1), y=data[:, i], color='blue')
		plt.xlabel('Metric')
		plt.ylabel('Loading')
		x_labels = ['NC', 'FFC', '1-FBC', 'FCF', '1/(ln(CSSD)+1)', 'FMCN', '1/(ACVF+1)', 'FPCF',
		            '1/(ACVC_NUC+1)', 'FPCC_NUC', 'AS_NUC', '1/(ACVC_CEN+1)', 'FPCC_CEN', 'AS_CEN']
		
		plt.xticks(np.arange(0, len(x_labels)), labels=x_labels, rotation=45, ha='right', rotation_mode='anchor')
		plt.savefig(join(save_dir, 'loading_vs_metric_' + PC[i] + '.png'), bbox_inches='tight', dpi=300)
		plt.clf()
		plt.close()


if __name__ == '__main__':
	
	compartment = sys.argv[2]
	# print(sys.argv[2])
	if sys.argv[1] == 'merge':
		noise_type = ['gaussian', 'downsampling']
	else:
		noise_type = [sys.argv[1]]
	tissue = sys.argv[4]
	if sys.argv[3] == 'all':
		modalities = ['CODEX', 'MIBI', 'CellDIVE', 'IMC']
	# modalities = ['CODEX']
	elif sys.argv[3] == 'both':
		modalities = ['CODEX', 'IMC']
	else:
		modalities = [sys.argv[3]]
	# if len(modalities) == 4:
	# 	modality_type = 'all'
	# else:
	# 	modality_type = modalities[0]
	# print(modality_type)
	modality_type = sys.argv[3]
	# print(modality_type)
	
	methods = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	           'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new',
	           'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
	
	methods_abre_nonrepaired = ['DeepCell 0.12.3 cell membrane', 'DeepCell 0.12.3 cytoplasm',
	                            'DeepCell 0.9.0 cell membrane',
	                            'DeepCell 0.9.0 cytoplasm', 'DeepCell 0.6.0 cell membrane', 'DeepCell 0.6.0 cytoplasm',
	                            'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
	                            'AICS(classic)',
	                            'Cellsegm', 'Voronoi']
	
	methods_abre_repaired = ['DeepCell 0.12.3 cell membrane+r', 'DeepCell 0.12.3 cytoplasm+r',
	                         'DeepCell 0.9.0 cell membrane+r',
	                         'DeepCell 0.9.0 cytoplasm+r', 'DeepCell 0.6.0 cell membrane+r',
	                         'DeepCell 0.6.0 cytoplasm+r',
	                         'Cellpose 2.1.0+r', 'Cellpose 0.6.1+r', 'Cellpose 0.0.3.1+r', 'CellProfiler+r', 'CellX+r',
	                         'AICS(classic)+r',
	                         'Cellsegm+r', 'Voronoi+r']
	
	methods_abre = methods_abre_nonrepaired + methods_abre_repaired
	
	method_num = len(methods_abre)
	noise_num = 3
	features_all = {}
	features_3D_all_modalities = {}
	features_2D_stack_pieces = []
	
	for noise in noise_type:
		data_dir = '/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/' + noise
		features_all[noise] = {}
		
		for modality in modalities:
			fig_dir = join(data_dir, 'nonrepaired', modality, tissue, compartment)
			features_3D = np.load(join(fig_dir, 'feature_3D.npy'))
			fig_dir = join(data_dir, 'repaired', modality, tissue, compartment)
			features_3D = np.hstack((features_3D, np.load(join(fig_dir, 'feature_3D.npy'))))
			features_all[noise][modality] = {}
			features_all[noise][modality]['data_3D'] = features_3D
			features_all[noise][modality]['data_2D'] = get_stack_features(features_3D)
			features_all[noise][modality]['num'] = int(features_3D.shape[0])
			features_2D_stack_pieces.append(features_all[noise][modality]['data_2D'])
			
	features_2D_stack = np.vstack(features_2D_stack_pieces)
	features_2D_stack = np.delete(features_2D_stack, 5, axis=-1)
	get_PCA_features(features_2D_stack)






