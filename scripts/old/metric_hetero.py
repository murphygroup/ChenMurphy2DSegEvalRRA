import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from os.path import join
import pandas as pd
from sklearn.preprocessing import normalize, scale, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

if __name__ == '__main__':
	modalities = ['CODEX', 'MIBI', 'CellDIVE', 'IMC']
	tissue_type = 'All'
	compartment = 'concatenated_compartments'


	for repaired_type in ['repaired', 'nonrepaired']:
		for noise_type in ['gaussian']:
			data_dir = '/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/' + noise_type + '/' + repaired_type
			save_dir = '/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/'
			metric_vs_metric_clustered_pieces = []
			img_names = []
			feature_3D_all_modalities = []
			img_num = 0
			for modality in modalities:
				feature_dir = join(data_dir, modality, tissue_type, compartment)
				feature_3D_modality = np.load(join(feature_dir, 'feature_3D.npy'))
				img_num_current = int(feature_3D_modality.shape[0] / 4)
				img_num += img_num_current
				feature_3D_modality_avg = np.average(feature_3D_modality[:img_num_current], axis=0)
				feature_3D_all_modalities.append(feature_3D_modality_avg)
			# print(feature_3D_noise[0].shape)
			# print(feature_3D_noise[1].shape)
			# np.stack((feature_3D_noise[0], feature_3D_noise[1]), axis=2)

			feature_3D_all_modalities = np.stack(feature_3D_all_modalities, axis=0)
			feature_3D_all_modalities_avg = np.average(feature_3D_all_modalities, axis=0)
			# print(feature_3D_all_modalities_avg.shape)
			# feature_3D_all_modalities_avg = feature_3D_all_modalities_avg / img_num
			
			
			

		
		
		
		
			save_dir = '/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/'
			metric_num = 14
			methods_abre = ['DeepCell 0.12.3 cell membrane', 'DeepCell 0.12.3 cytoplasm',
			                            'DeepCell 0.9.0 cell membrane',
			                            'DeepCell 0.9.0 cytoplasm', 'DeepCell 0.6.0 cell membrane',
			                            'DeepCell 0.6.0 cytoplasm',
			                            'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
			                            'AICS(classic)',
			                            'Cellsegm', 'Voronoi']
			# methods_abre = [ 'DeepCell 0.6.0 cell membrane',
			#                 'DeepCell 0.6.0 cytoplasm', 'DeepCell 0.9.0 cell membrane', 'DeepCell 0.9.0 cytoplasm',
			#                  'Cellpose 0.0.3.1', 'Cellpose 0.6.1', 'CellProfiler', 'CellX', 'AICS(classic)',
			#                 'Cellsegm', 'Voronoi']
			metric_abre = ['NC', 'FFC', '1-FBC', 'FCF', '1/(ln(CSSD)+1)', 'FMCN', '1/(ACVF+1)', 'FPCF',
			               '1/(ACVC_NUC+1)', 'FPCC_NUC', 'AS_NUC',
			               '1/(ACVC_CEN+1)', 'FPCC_CEN', 'AS_CEN']
			
			noise_type = 'gaussian'
			data_dir = '/home/hrchen/Documents/Research/hubmap/fig/manuscript_v1.8/' + noise_type + '/' + repaired_type
			
			# avg_metric_vs_metric_clustered_stack_df = pd.read_csv(
			# 	join(data_dir, 'all_modalities', compartment, 'avg_metric_vs_metric_all_modalities.csv'), index_col=0)
			# avg_metric_vs_metric_clustered_stack_df = avg_metric_vs_metric_clustered_stack_df.reindex(methods_abre)
			# avg_metric_vs_metric_clustered_stack = avg_metric_vs_metric_clustered_stack_df.values
			avg_metric_vs_metric_clustered_stack = feature_3D_all_modalities_avg
			# cmap = ['deepskyblue', 'darkviolet', 'darkred', 'deeppink',
			#          'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate',
			#         'darkgoldenrod', 'darkcyan']
			cmap_repaired = ['darkorchid', 'darkturquoise', 'deepskyblue', 'darkviolet', 'darkred', 'deeppink',
			                 'seagreen', 'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate',
			                 'darkgoldenrod', 'darkcyan']
			
			for i in range(len(methods_abre)):
				# sns.lineplot(x=np.arange(1, metric_num+1), y=avg_metric_vs_metric_clustered_stack[i, :], sort=False, label=methods_abre[i], color=cmap[i])
				sns.lineplot(x=np.arange(1, metric_num + 1), y=avg_metric_vs_metric_clustered_stack[i, :], sort=False,
				             label='_nolegend_', color=cmap[i])
			
			plt.xticks(np.arange(1, metric_num + 1), labels=metric_abre, rotation=45, rotation_mode='anchor', ha='right',
			           fontsize=9)
			# plt.set_xticklabels(metric_abre)
			plt.xlabel("Metric")
			plt.ylabel("Average Metric Value")
			# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), fancybox=True, shadow=True, ncol=3)
			plt.ylim(-0.8, 1.05)
			avg_metric_vs_metric_clustered_stack_df = pd.DataFrame(data=avg_metric_vs_metric_clustered_stack, dtype=float,
			                                                       index=methods_abre)
			# avg_metric_vs_metric_clustered_stack_df.to_csv(join(save_dir, 'avg_metric_vs_metric_all_modalities.csv'))
			plt.savefig(join(save_dir, 'avg_metric_vs_metric_all_modalities_' + repaired_type + '_test.png'), bbox_inches='tight',
			            dpi=400)
			plt.clf()