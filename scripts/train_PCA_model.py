import bz2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from os.path import join
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.markers import MarkerStyle
import pandas as pd
import pickle
import matplotlib.path as mpath
from matplotlib import gridspec


def visualization_pc_all_img(feature_matrix):
	# print(feature_matrix)
	
	if neg_flag_PC1 == True:
		pc1 = feature_matrix[:, 0]
	else:
		pc1 = feature_matrix[:, 0]
	
	if neg_flag_PC2 == True:
		pc2 = -feature_matrix[:, 1]
	else:
		pc2 = feature_matrix[:, 1]
	

	# plt.scatter(pc1, pc2, color=cmap[i],
	# 	            s=20, label='_no_legend_')
	
	plt.scatter(pc1, pc2, color=cmap[i],
		            s=20, label=methods_abre[i])

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


	if method_type == 'grey':
		if i not in less_methods:
			
			ax2.scatter(avg_pc1, avg_pc2, color=cmap[i],
			            s=50, marker=marker[i], label='_no_legend_')
		else:
			ax2.scatter(avg_pc1, avg_pc2, color=cmap[i],
			            s=50, marker=marker[i], label='_no_legend_')
	else:
		ax2.scatter(avg_pc1, avg_pc2, color=cmap[i],
			            s=50, marker = marker[i], label=methods_abre[i])
		# plt.scatter(avg_pc1, avg_pc2, color=cmap[i],
		# 	            s=50, marker = marker[i], label='_no_legend_')
	# print(feature_matrix[0, 0], '*', loading[0], '+', feature_matrix[0, 1], '*', loading[1])
	# return avg_pc1 * loading[0] + avg_pc2 * loading[1]
	return avg_pc1 * loading[0] + avg_pc2 * loading[1]



def visualization_track(track, method_num):
	if neg_flag_PC1 == True:
		track[:, 0] = -track[:, 0]
	
	if neg_flag_PC2 == True:
		track[:, 1] = -track[:, 1]
		
	ax2.plot(track[:, 0], track[:, 1], color=cmap[method_num])
	ax2.scatter(track[:, 0], track[:, 1], color=cmap[method_num],  s=10)





def get_stack_features(features):
	feature_2d = np.empty((0, features.shape[2]))
	for i in range(method_num):
		feature_method = features[:, i, :]
		feature_2d = np.vstack((feature_2d, feature_method))
	return feature_2d




def get_PCA_features(features, dir):

	if modality_type == 'all_modalities' and len(noise_type) > 1:
		ss = StandardScaler().fit(features)

		feature_2d_scaled = ss.transform(features)

		pca = PCA(n_components=2).fit(feature_2d_scaled)
		for n_com in range(2):
			if np.sum(pca.components_.T[:, n_com]) < 0:
				pca.components_.T[:, n_com] = -pca.components_.T[:, n_com]

		pickle.dump([ss, pca], open(join(dir, 'pca.pickle'), "wb"))

	else:
		
		pca_model = pickle.load(open(join(dir, 'pca.pickle'), 'rb'))
		ss = pca_model[0]
		pca = pca_model[1]
		feature_2d_scaled = ss.transform(features)
	feature_2d_PCA = pca.transform(feature_2d_scaled)
	return feature_2d_PCA, pca, ss

def visualization_loadings(data, dir):
	PC = ['PC1', 'PC2']
	for i in range(2):
		sns.barplot(x=np.arange(1, data.shape[0]+1), y=data[:, i], color='blue')
		plt.xlabel('Metric', fontsize=15)
		plt.ylabel('Loading', fontsize=15)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)
		x_labels = ['NC', 'FFC', '1-FBC', 'FCF', '1/(ln(CSSD)+1)', 'FMCN', '1/(ACVF+1)', 'FPCF',
		            '1/(ACVC_NUC+1)', 'FPCC_NUC', 'AS_NUC','1/(ACVC_CEN+1)', 'FPCC_CEN', 'AS_CEN']
		plt.xticks(np.arange(0, len(x_labels) ), labels=x_labels, rotation=45, ha='right', rotation_mode='anchor')
		plt.savefig(join(dir, 'loading_vs_metric_'+ PC[i] + '.png'), bbox_inches='tight', dpi=300)
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
	if sys.argv[3] == 'all_modalities':
		modalities = ['CODEX', 'MIBI', 'CellDIVE', 'IMC']
		# modalities = ['CODEX']
	elif sys.argv[3] == 'CODEX_IMC':
		modalities = ['CODEX', 'IMC']
	else:
		modalities = [sys.argv[3]]

	modality_type = sys.argv[3]
	method_type = sys.argv[5]

	methods = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	           'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new',
	           'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
	
	# methods_abre_nonrepaired = ['DeepCell 0.12.3 cell membrane', 'DeepCell 0.12.3 cytoplasm', 'DeepCell 0.9.0 cell membrane',
	#                 'DeepCell 0.9.0 cytoplasm', 'DeepCell 0.6.0 cell membrane', 'DeepCell 0.6.0 cytoplasm',
	#                 'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX', 'AICS(classic)',
	#                 'Cellsegm', 'Voronoi']
	#
	# methods_abre_repaired = ['DeepCell 0.12.3 cell membrane+r', 'DeepCell 0.12.3 cytoplasm+r', 'DeepCell 0.9.0 cell membrane+r',
	#                 'DeepCell 0.9.0 cytoplasm+r', 'DeepCell 0.6.0 mem+r', 'DeepCell 0.6.0 cytoplasm+r',
	#                 'Cellpose 2.1.0+r', 'Cellpose 0.6.1+r', 'Cellpose 0.0.3.1+r', 'CellProfiler+r', 'CellX+r', 'AICS(classic)+r',
	#                 'Cellsegm+r', 'Voronoi+r']

	methods_abre_nonrepaired = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto', 'DeepCell 0.9.0 mem',
	                'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX', 'AICS(classic)',
	                'Cellsegm', 'Voronoi']

	methods_abre_repaired = ['DeepCell 0.12.3 mem+r', 'DeepCell 0.12.3 cyto+r', 'DeepCell 0.9.0 mem+r',
	                'DeepCell 0.9.0 cyto+r', 'DeepCell 0.6.0 mem+r', 'DeepCell 0.6.0 cyto+r',
	                'Cellpose 2.1.0+r', 'Cellpose 0.6.1+r', 'Cellpose 0.0.3.1+r', 'CellProfiler+r', 'CellX+r', 'AICS(classic)+r',
	                'Cellsegm+r', 'Voronoi+r']
	
	methods_abre = methods_abre_nonrepaired + methods_abre_repaired

	method_num = len(methods_abre)
	noise_num = 3
	features_all = {}
	features_3D_all_modalities = {}
	features_2D_stack_pieces = []
	
	file_dir = os.path.dirname(os.getcwd())
	for noise in noise_type:
		data_dir = join(file_dir, 'data', 'metrics', noise)
		features_all[noise] = {}
		
		for modality in modalities:
			feature_dir = join(data_dir, 'nonrepaired', modality, tissue, compartment)
			features_3D = pickle.load(bz2.BZ2File(join(feature_dir, 'feature_3D.pickle'), 'r'))
			feature_dir = join(data_dir, 'repaired', modality, tissue, compartment)
			features_3D = np.hstack((features_3D, pickle.load(bz2.BZ2File(join(feature_dir, 'feature_3D.pickle'), 'r'))))
			# features_3D[:, :, 8] = (features_3D[:, :, 8] + features_3D[:, :, 11]) / 2
			# features_3D[:, :, 9] = (features_3D[:, :, 9] + features_3D[:, :, 12]) / 2
			# features_3D[:, :, 10] = (features_3D[:, :, 10] + features_3D[:, :, 13]) / 2
			# features_3D = np.delete(features_3D, 13, axis=2)
			# features_3D = np.delete(features_3D, 12, axis=2)
			# features_3D = np.delete(features_3D, 11, axis=2)
			# features_3D[:, 26, :] = 0
			# features_3D[:, 26, -1] = -1
			# features_3D[:, 25, :] = 0
			# features_3D[:, 25, -1] = -1
			# features_3D[:, 24, :] = 0
			# features_3D[:, 24, -1] = -1
			# features_3D[:, 23, :] = 0
			# features_3D[:, 22, :] = 0
			# print(features_3D.shape)
			if modality == 'CODEX':
				test = pickle.load(bz2.BZ2File(join(feature_dir, 'feature_3D.pickle'), 'r'))
				idx = 0
				test1 = test[idx, 0]
				# test2 = test[idx, 10]
				# print(test.shape)
				# print(test[idx, 0])
				# print(test[idx, 10])
			features_all[noise][modality] = {}
			features_all[noise][modality]['data_3D'] = features_3D
			features_all[noise][modality]['data_2D'] = get_stack_features(features_3D)
			features_all[noise][modality]['num'] = int(features_3D.shape[0])
			features_2D_stack_pieces.append(features_all[noise][modality]['data_2D'])
			

	features_2D_stack = np.vstack(features_2D_stack_pieces)
	
	output_dir = join(file_dir, 'data', 'output')
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	features_2D_stack_PCA, PCA_model, ss_model = get_PCA_features(features_2D_stack, output_dir)
	
	test1 = np.expand_dims(test1, 0)
	print(PCA_model.transform(ss_model.transform(test1)))
	# test2 = np.expand_dims(test2, 0)
	# print(PCA_model.transform(ss_model.transform(test2)))
	start_index = 0
	for noise in noise_type:
		for modality in modalities:
			# print(features_all['gaussian']['CODEX'])
			# print(features_all[noise][modality]['num'])
			features_all[noise][modality]['data_2D_PCA'] = features_2D_stack_PCA[start_index:(start_index + features_all[noise][modality]['num'] * method_num), :]
			start_index = start_index + features_all[noise][modality]['num'] * method_num
		
	# visualization
	neg_flag_PC1 = False
	neg_flag_PC2 = False
	loadings = pd.DataFrame(PCA_model.components_.T, columns=['PC1', 'PC2'])

	if loadings.values[0, 0] < 0:
		loadings.values[:, 0] = -loadings.values[:, 0]
		neg_flag_PC1 = True
	if loadings.values[0, 1] < 0:
		loadings.values[:, 1] = -loadings.values[:, 1]
		neg_flag_PC2 = True

	figure_dir = join(file_dir, 'figures')
	


	if method_type == 'all':
		if len(noise_type) > 1:
			visualization_loadings(loadings.values, figure_dir)
	
	star = mpath.Path.unit_regular_star(6)
	
	marker_nonrepaired = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o"]
	marker_repaired = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o"]
	marker = marker_nonrepaired + marker_repaired
	
	track = np.empty((noise_num + 1, method_num, 2))
	less_methods = [0, 1, 6, 9, 10, 11, 12, 13, 14, 15, 20, 23, 24, 25, 26, 27]
	
	if method_type == 'grey':
		method_range = range(method_num)
		cmap_repaired = ['lightgrey', 'lightgrey', 'deepskyblue', 'darkviolet', 'darkred', 'deeppink', 'lightgrey', 'darkgreen', 'darkkhaki', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey']
		cmap_nonrepaired = ['lightgrey', 'lightgrey', 'skyblue', 'violet', 'red', 'lightpink', 'lightgrey', 'limegreen', 'khaki', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey']
		cmap = cmap_nonrepaired + cmap_repaired
	elif method_type == 'less':
		method_range = less_methods
		cmap_repaired = ['darkorchid', 'darkturquoise', 'deepskyblue', 'darkviolet', 'darkred', 'deeppink', 'seagreen', 'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan']
		cmap_nonrepaired = ['orchid', 'turquoise', 'skyblue', 'violet', 'red', 'lightpink', 'darkseagreen', 'limegreen', 'khaki', 'slateblue', 'lightsalmon', 'sandybrown', 'goldenrod', 'cyan']
		cmap = cmap_nonrepaired + cmap_repaired
	else:
		method_range = range(method_num)
		cmap_repaired = ['darkorchid', 'darkturquoise', 'deepskyblue', 'darkviolet', 'darkred', 'deeppink', 'seagreen', 'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan']
		cmap_nonrepaired = ['orchid', 'turquoise', 'skyblue', 'violet', 'red', 'lightpink', 'darkseagreen', 'limegreen', 'khaki', 'slateblue', 'lightsalmon', 'sandybrown', 'goldenrod', 'cyan']
		cmap = cmap_nonrepaired + cmap_repaired
	

	
	for noise in noise_type:
		if method_type == 'tissue' or method_type == 'modality':
			fig = plt.figure()
			fig.set_figheight(5)
			fig.set_figwidth(15)
			
			gs = gridspec.GridSpec(20, 20, left=0.7, right=1.6, top=2, hspace=0.25, wspace=4)
			ax1 = fig.add_subplot(gs[:, :10])
			ax2 = fig.add_subplot(gs[:-9, 11:])
		
		else:
			fig, ax2 = plt.subplots()
		pca_combined = []
		features_2D_stack_PCA_all_modalities_method_noise_combined = []
		for noise_level in range(noise_num + 1):
			for i in method_range:
				features_2D_stack_PCA_all_modalities_method_noise = np.empty((0, 2))
				for modality in modalities:
					noise_img_num = features_all[noise][modality]['num'] / (noise_num + 1)
					features_2D_stack_PCA_modality_method_all_noise = features_all[noise][modality]['data_2D_PCA'][features_all[noise][modality]['num']*i:features_all[noise][modality]['num']*(i+1),:]
					features_2D_stack_PCA_modality_method_noise = features_2D_stack_PCA_modality_method_all_noise[int(noise_img_num*noise_level):int(noise_img_num*(noise_level+1)),:]
	
					features_2D_stack_PCA_all_modalities_method_noise = np.vstack((features_2D_stack_PCA_all_modalities_method_noise, np.average(features_2D_stack_PCA_modality_method_noise, axis=0)))
				
				track[noise_level, i, :] = np.average(features_2D_stack_PCA_all_modalities_method_noise, axis=0)

				if noise_level == 0:
					features_2D_stack_PCA_all_modalities_method_noise_combined.append(features_2D_stack_PCA_all_modalities_method_noise)
			
		
		for i in method_range:
			visualization_track(track[:, i, :], i)
		features_2D_stack_PCA_all_modalities_method_noise_combined = np.stack(features_2D_stack_PCA_all_modalities_method_noise_combined, axis=0)
		idx = 0
		for i in method_range:
			pca_combined.append(visualization_avg(features_2D_stack_PCA_all_modalities_method_noise_combined[idx],
			                                      PCA_model.explained_variance_ratio_))
			idx += 1
			
		if method_type == 'tissue' or method_type == 'modality':
			

			
			ax2.set_xlabel('PC1 (' + str(round(PCA_model.explained_variance_ratio_[0] * 100)) + '%)', fontsize=18)
			ax2.set_ylabel('PC2 (' + str(round(PCA_model.explained_variance_ratio_[1] * 100)) + '%)', fontsize=18)
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=15)
			ax2.tick_params(axis='both', which='major', labelsize=18)

			df = pd.DataFrame({"Method": methods_abre, "Quality_Score": pca_combined})
			df_sorted = df.sort_values('Quality_Score', ascending=False)
			
			barlist = ax1.barh('Method', 'Quality_Score', data=df_sorted)
			
			ax1.invert_yaxis()  # labels read top-to-bottom
			ax1.set_xlabel('Quality Score', fontsize=18)
			ax1.tick_params(axis='both', which='major', labelsize=18)
			if method_type == 'tissue':
				plt.savefig(join(figure_dir, 'combined_PC_and_quality_score_rankings_' + sys.argv[4] + '.png'), bbox_inches='tight', dpi=300)
			elif method_type == 'modality':
				plt.savefig(join(figure_dir, 'combined_PC_and_quality_score_rankings_' + sys.argv[3] + '.png'), bbox_inches='tight', dpi=300)
			plt.clf()
			plt.close()
			
		else:
			
			plt.xlabel('PC1 (' + str(round(PCA_model.explained_variance_ratio_[0] * 100)) + '%)', fontsize=15)
			plt.ylabel('PC2 (' + str(round(PCA_model.explained_variance_ratio_[1] * 100)) + '%)', fontsize=15)
			# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
			plt.yticks(fontsize=15)
			plt.xticks(fontsize=15)
			plt.savefig(join(figure_dir, 'avg_top2PC_all_modalities_' + noise + '_' + method_type + '.png'),
			            bbox_inches='tight', dpi=300)
			plt.clf()
			plt.close()

		
		if method_type == 'all' and noise == 'gaussian':
			df = pd.DataFrame({"Method": methods_abre, "Quality_Score": pca_combined})
			sns.barplot(x='Quality_Score', y='Method', data=df,
			            order=df.sort_values('Quality_Score', ascending=False).Method,
			            palette=["b" if y != 'Voronoi+r' else 'r' for y in
			                     df.sort_values('Quality_Score', ascending=False).Method])
			plt.ylabel('Method')
			plt.xlabel('Quality Score')
			# plt.xscale("log")
			# from matplotlib.ticker import ScalarFormatter
			#
			# plt.gca().xaxis.set_major_formatter(ScalarFormatter())
			plt.xlim(min(df['Quality_Score'])-0.1, max(df['Quality_Score'])+0.1)
			plt.savefig(join(figure_dir, 'quality_score_rankings.png'), bbox_inches='tight', dpi=300)
			plt.clf()
			plt.close()
		
			pca_combined = []
			for noise_level in [0]:
				for i in method_range:
					features_2D_stack_PCA_all_modalities_method_noise = np.empty((0, 2))
					for modality in modalities:
						noise_img_num = features_all[noise][modality]['num'] / (noise_num + 1)
						features_2D_stack_PCA_modality_method_all_noise = features_all[noise][modality][
							                                                  'data_2D_PCA'][
						                                                  features_all[noise][modality]['num'] * i:
						                                                  features_all[noise][modality]['num'] * (
								                                                  i + 1), :]
						features_2D_stack_PCA_modality_method_noise = features_2D_stack_PCA_modality_method_all_noise[
						                                              int(noise_img_num * noise_level):int(
							                                              noise_img_num * (noise_level + 1)), :]
						
						features_2D_stack_PCA_all_modalities_method_noise = np.vstack((
							features_2D_stack_PCA_all_modalities_method_noise,
							
							features_2D_stack_PCA_modality_method_noise
						))
					print(features_2D_stack_PCA_all_modalities_method_noise.shape)
					if noise_level == 0:
						pca_combined.append(
							visualization_pc_all_img(features_2D_stack_PCA_all_modalities_method_noise))
				
				plt.xlabel('PC1 (' + str(round(PCA_model.explained_variance_ratio_[0] * 100)) + '%)', fontsize=18)
				plt.ylabel('PC2 (' + str(round(PCA_model.explained_variance_ratio_[1] * 100)) + '%)', fontsize=18)
				plt.yticks(fontsize=18)
				plt.xticks(fontsize=18)
				# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
				
				plt.savefig(join(figure_dir, 'top2PC_scatter_all_modalities.png'),
				            bbox_inches='tight', dpi=300)
				plt.clf()
				plt.close()
	

