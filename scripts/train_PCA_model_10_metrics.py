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
import bz2
import pickle


def get_stack_features(features):
	feature_2d = np.empty((0, features.shape[2]))
	for i in range(method_num):
		feature_method = features[:, i, :]
		feature_2d = np.vstack((feature_2d, feature_method))
	return feature_2d


def get_PCA_features(features, dir):

	if modality_type == 'all' and len(noise_type) > 1:
		ss = StandardScaler().fit(features)
		
		feature_2d_scaled = ss.transform(features)
		
		pca = PCA(n_components=2).fit(feature_2d_scaled)
		for n_com in range(2):
			if np.sum(pca.components_.T[:, n_com]) < 0:
				pca.components_.T[:, n_com] = -pca.components_.T[:, n_com]
		# print(pca.components_.T)
		# print(pca.explained_variance_ratio_)
		pickle.dump([ss, pca], open(join(dir, 'pca_10_metrics.pickle'), "wb"))




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
	
	file_dir = os.getcwd()

	for noise in noise_type:
		data_dir = join(file_dir, 'data', 'metrics', noise)
		features_all[noise] = {}
		
		for modality in modalities:
			feature_dir = join(data_dir, 'nonrepaired', modality, tissue, compartment)
			features_3D = pickle.load(bz2.BZ2File(join(feature_dir, 'feature_3D.pickle'), 'r'))
			feature_dir = join(data_dir, 'repaired', modality, tissue, compartment)
			features_3D = np.hstack((features_3D, pickle.load(bz2.BZ2File(join(feature_dir, 'feature_3D.pickle'), 'r'))))
			features_all[noise][modality] = {}
			features_all[noise][modality]['data_3D'] = features_3D
			features_all[noise][modality]['data_2D'] = get_stack_features(features_3D)
			features_all[noise][modality]['num'] = int(features_3D.shape[0])
			features_2D_stack_pieces.append(features_all[noise][modality]['data_2D'])
			
	features_2D_stack = np.vstack(features_2D_stack_pieces)
	features_2D_stack = np.delete(features_2D_stack, 5, axis=-1)
	# print(features_2D_stack.shape)
	output_dir = join(file_dir, 'data', 'output')
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	get_PCA_features(features_2D_stack, output_dir)






