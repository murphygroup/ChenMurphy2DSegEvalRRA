import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from os.path import join
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import pickle
from matplotlib import gridspec
import matplotlib.path as mpath


def visualization_avg(feature_matrix, loading):
	
	if neg_flag_PC1 == True:
		avg_pc1 = -np.average(feature_matrix[:, 0])
	else:
		avg_pc1 = np.average(feature_matrix[:, 0])
	
	if neg_flag_PC2 == True:
		avg_pc2 = -np.average(feature_matrix[:, 1])
	else:
		avg_pc2 = np.average(feature_matrix[:, 1])

	ax2.scatter(avg_pc1, avg_pc2, color=cmap[i],
		            s=50, marker = marker[i], label=methods_abre[i])
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


def get_PCA_features(features):
	print(features.shape)
	save_dir = join('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8', 'merge', 'all_modalities', compartment, modality_type, tissue)
	if modality_type == 'all' and len(noise_type) > 1:
		ss = StandardScaler().fit(features)

		feature_2d_scaled = ss.transform(features)

		pca = PCA(n_components=2).fit(feature_2d_scaled)
		for n_com in range(2):
			if np.sum(pca.components_.T[:, n_com]) < 0:
				pca.components_.T[:, n_com] = -pca.components_.T[:, n_com]
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		pickle.dump([ss, pca], open(join(save_dir, 'pca.pickle'), "wb"))

	else:
		
		pca_model = pickle.load(open(join(os.path.dirname(os.path.dirname(save_dir)), 'all/All/pca.pickle'), 'rb'))
		ss = pca_model[0]
		pca = pca_model[1]
		feature_2d_scaled = ss.transform(features)
	feature_2d_PCA = pca.transform(feature_2d_scaled)
	# print(feature_2d_PCA)
	return feature_2d_PCA, pca, ss

def visualization_loadings(data):
	PC = ['PC1', 'PC2']
	for i in range(2):
		sns.barplot(x=np.arange(1, data.shape[0]+1), y=data[:, i], color='blue')
		plt.xlabel('Metric')
		plt.ylabel('Loading')
		x_labels = ['NC', 'FFC', '1-FBC', 'FCF', '1/(ln(CSSD)+1)', 'FMCN', '1/(ACVF+1)', 'FPCF',
		            '1/(ACVC_NUC+1)', 'FPCC_NUC', 'AS_NUC','1/(ACVC_CEN+1)', 'FPCC_CEN', 'AS_CEN']

		plt.xticks(np.arange(0, len(x_labels) ), labels=x_labels, rotation=45, ha='right', rotation_mode='anchor')
		plt.savefig(join(save_dir, 'loading_vs_metric_'+ PC[i] + '.png'), bbox_inches='tight', dpi=300)
		plt.clf()
		plt.close()


if __name__ == '__main__':
	
	if sys.argv[1] == 'merge':
		noise_type = ['gaussian', 'downsampling']
	else:
		noise_type = [sys.argv[1]]

	compartment = sys.argv[2]

	if sys.argv[3] == 'all':
		modalities = ['CODEX', 'MIBI', 'CellDIVE', 'IMC']
	elif sys.argv[3] == 'both':
		modalities = ['CODEX', 'IMC']
	else:
		modalities = [sys.argv[3]]
	modality_type = sys.argv[3]
	
	tissue = sys.argv[4]


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
	features_2D_stack_PCA, PCA_model, ss_model = get_PCA_features(features_2D_stack)


	start_index = 0
	for noise in noise_type:
		for modality in modalities:
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
	
	if len(noise_type) > 1:
		save_dir = join('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8', 'merge', 'all_modalities', compartment, modality_type, tissue)
		visualization_loadings(loadings.values)
	


	star = mpath.Path.unit_regular_star(6)


	
	track = np.empty((noise_num + 1, method_num, 2))
	
	fig = plt.figure()
	fig.set_figheight(5)
	fig.set_figwidth(15)
	
	
	gs = gridspec.GridSpec(10, 10, left=0.35, right=1.1, top=1.3, hspace=0.25, wspace=4)
	ax1 = fig.add_subplot(gs[:, :5])
	ax2 = fig.add_subplot(gs[:-4, 5:])
	
	for noise in noise_type:
		save_dir = join('/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8', noise, 'all_modalities', compartment, modality_type, tissue)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		pca_combined = []
		features_2D_stack_PCA_all_modalities_method_noise_combined = []

		for noise_level in range(noise_num + 1):
			for i in range(method_num):
				features_2D_stack_PCA_all_modalities_method_noise = np.empty((0, 2))
				for modality in modalities:
					noise_img_num = features_all[noise][modality]['num'] / (noise_num + 1)
					features_2D_stack_PCA_modality_method_all_noise = features_all[noise][modality]['data_2D_PCA'][features_all[noise][modality]['num']*i:features_all[noise][modality]['num']*(i+1),:]
					features_2D_stack_PCA_modality_method_noise = features_2D_stack_PCA_modality_method_all_noise[int(noise_img_num*noise_level):int(noise_img_num*(noise_level+1)),:]
	
					features_2D_stack_PCA_all_modalities_method_noise = np.vstack((features_2D_stack_PCA_all_modalities_method_noise, np.average(features_2D_stack_PCA_modality_method_noise, axis=0)))
					
				if noise_level == 0:
					features_2D_stack_PCA_all_modalities_method_noise_combined.append(features_2D_stack_PCA_all_modalities_method_noise)
				
				track[noise_level, i, :] = np.average(features_2D_stack_PCA_all_modalities_method_noise, axis=0)
			
	
		for i in range(method_num):
			visualization_track(track[:, i, :], i)
		features_2D_stack_PCA_all_modalities_method_noise_combined = np.stack(features_2D_stack_PCA_all_modalities_method_noise_combined, axis=0)
		idx = 0
		for i in range(method_num):
			pca_combined.append(visualization_avg(features_2D_stack_PCA_all_modalities_method_noise_combined[idx],
			                                      PCA_model.explained_variance_ratio_))
			idx += 1
	
		ax2.set_xlabel('PC1 (' + str(PCA_model.explained_variance_ratio_[0])[2:4] + '%)')
		if str(PCA_model.explained_variance_ratio_[1])[2] == '0':
			ax2.set_ylabel('PC2 (' + str(PCA_model.explained_variance_ratio_[1])[3:4] + '%)')
		else:
			ax2.set_ylabel('PC2 (' + str(PCA_model.explained_variance_ratio_[1])[2:4] + '%)')
		ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
		
		df = pd.DataFrame({"Method":methods_abre, "Quality_Score":pca_combined})
		df_sorted = df.sort_values('Quality_Score', ascending=False)
		barlist = ax1.barh('Method', 'Quality_Score', data=df_sorted)
		ax1.invert_yaxis()  # labels read top-to-bottom
		ax1.set_xlabel('Quality Score')
		plt.savefig(join(save_dir, 'combined_PC_quality_score.png'), bbox_inches='tight', dpi=300)
		plt.clf()
		plt.close()
	



