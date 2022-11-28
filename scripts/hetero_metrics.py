import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import bz2
import pickle
from os.path import join
import pandas as pd
from sklearn.preprocessing import normalize, scale, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

if __name__ == '__main__':
	modalities = ['CODEX', 'MIBI', 'CellDIVE', 'IMC']
	tissue_type = 'all_tissues'
	compartment = 'concatenated_compartments'

	file_dir = os.path.dirname(os.getcwd())
	print(os.getcwd())
	output_dir = join(file_dir, 'data', 'output')
	pca_model = pickle.load(open(join(output_dir, 'pca.pickle'), 'rb'))
	ss = pca_model[0]
	pca = pca_model[1]
	
	for repaired_type in ['repaired', 'nonrepaired']:
		for noise_type in ['gaussian']:
			data_dir = join(file_dir, 'data', 'metrics', noise_type, repaired_type)
			save_dir = join(file_dir, 'figures')
			metric_vs_metric_clustered_pieces = []
			img_names = []
			feature_3D_all_modalities = []
			img_num = 0
			for modality in modalities:
				feature_dir = join(data_dir, modality, tissue_type, compartment)
				feature_3D_modality = pickle.load(bz2.BZ2File(join(feature_dir, 'feature_3D.pickle'), 'r'))
				for m in range(feature_3D_modality.shape[1]):
					feature_3D_modality[:, m, :] = ss.transform(feature_3D_modality[:, m, :])
				# print()
				img_num_current = int(feature_3D_modality.shape[0] / 4)
				img_num += img_num_current
				# print(feature_3D_modality[:img_num_current].shape)
				# print(feature_3D_modality[:img_num_current][:, -2, -1])
				# feature_3D_modality_avg = np.average(feature_3D_modality[:img_num_current], axis=0)
				# if modality == modalities[0]:
				# 	feature_3D_modality_avg = feature_3D_modality[:img_num_current]
				# else:
				# 	feature_3D_modality_avg += feature_3D_modality[:img_num_current]
				# feature_3D_modality_avg = feature_3D_modality_avg / img_num
				if modality == modalities[0]:
					feature_3D_all_modalities = np.sum(feature_3D_modality[:img_num_current], axis=0)
				else:
					feature_3D_all_modalities += np.sum(feature_3D_modality[:img_num_current], axis=0)
				# print(feature_3D_modality[:img_num_current].shape)
				# feature_3D_all_modalities.append(np.sum(feature_3D_modality[:img_num_current], axis=0))


			# feature_3D_all_modalities = np.stack(feature_3D_all_modalities)
			# feature_3D_all_modalities_avg = np.average(feature_3D_all_modalities, axis=0)
			feature_3D_all_modalities_avg = feature_3D_all_modalities / img_num

		
			methods_abre = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto', 'DeepCell 0.9.0 mem',
			                            'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
			                            'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
			                            'AICS(classic)',
			                            'Cellsegm', 'Voronoi']

			metric_abre = ['NC', 'FFC', '1-FBC', 'FCF', '1/(ln(CSSD)+1)', 'FMCN', '1/(ACVF+1)', 'FPCF',
			               '1/(ACVC_NUC+1)', 'FPCC_NUC', 'AS_NUC',
			               '1/(ACVC_CEN+1)', 'FPCC_CEN', 'AS_CEN']
			metric_num = len(metric_abre)
			noise_type = 'gaussian'
			
			# avg_metric_vs_metric_clustered_stack_df = pd.read_csv(
			# 	join(data_dir, 'all_modalities', compartment, 'avg_metric_vs_metric_all_modalities.csv'), index_col=0)
			# avg_metric_vs_metric_clustered_stack_df = avg_metric_vs_metric_clustered_stack_df.reindex(methods_abre)
			# avg_metric_vs_metric_clustered_stack = avg_metric_vs_metric_clustered_stack_df.values
			avg_metric_vs_metric_clustered_stack = feature_3D_all_modalities_avg
			# cmap = ['deepskyblue', 'darkviolet', 'darkred', 'deeppink',
			#          'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate',
			#         'darkgoldenrod', 'darkcyan']
			cmap = ['darkorchid', 'darkturquoise', 'deepskyblue', 'darkviolet', 'darkred', 'deeppink',
			                 'seagreen', 'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate',
			                 'darkgoldenrod', 'darkcyan']
			
			for i in range(len(methods_abre)):
				# sns.lineplot(x=np.arange(1, metric_num+1), y=avg_metric_vs_metric_clustered_stack[i, :], sort=False, label=methods_abre[i], color=cmap[i])
				sns.lineplot(x=np.arange(1, metric_num + 1), y=avg_metric_vs_metric_clustered_stack[i, :], sort=False,
				             label='_nolegend_', color=cmap[i])
			
			plt.xticks(np.arange(1, metric_num + 1), labels=metric_abre, rotation=45, rotation_mode='anchor', ha='right',
			           fontsize=13)
			plt.yticks(fontsize=13)
			# plt.set_xticklabels(metric_abre)
			plt.xlabel("Metric", fontsize=13)
			if repaired_type == 'nonrepaired':
				plt.ylabel("Average Metric Value", fontsize=13)
			if repaired_type == 'repaired':
				plt.yticks(color='w', fontsize=0)
			# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fancybox=True, shadow=True, ncol=3)
			plt.ylim(-2, 2.6)
			# plt.ylim(-0.8, 1.1)

			avg_metric_vs_metric_clustered_stack_df = pd.DataFrame(data=avg_metric_vs_metric_clustered_stack, dtype=float,
			                                                       index=methods_abre)
			plt.savefig(join(save_dir, 'avg_metric_vs_metric_all_modalities_' + repaired_type + '.png'), bbox_inches='tight',
			            dpi=400)
			plt.clf()