import json
import os.path

import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
import numpy as np
import glob

if __name__ == '__main__':
	# merged_list = ['DeepCell 0.12.3 mem 95% after merged', 'DeepCell 0.12.3 mem 90% after merged', 'DeepCell 0.12.3 mem 75% after merged', 'DeepCell 0.12.3 mem 60% after merged']
	mask_dir_list = sorted(glob.glob('/Volumes/Extreme/segmentation/CODEX/HBM**', recursive=True))
	quality_score_list_pieces = []
	metric_abre = ['NC', 'FFC', '1-FBC', 'FCF', '1/(ln(CSSD)+1)', 'FMCN', '1/(ACVF+1)', 'FPCF',
	               '1/(ACVC_NUC+1)', 'FPCC_NUC', 'AS_NUC',
	               '1/(ACVC_CEN+1)', 'FPCC_CEN', 'AS_CEN']
	metric_all = []
	for mask_dir in mask_dir_list:
		if mask_dir == '/Volumes/Extreme/segmentation/CODEX/HBM988.SNDW.698':
			mask_dir = join(mask_dir, 'R003_X004_Y004', 'random_gaussian_0', 'merged_fraction_mask')
		elif mask_dir == '/Volumes/Extreme/segmentation/CODEX/HBM433.MQRQ.278':
			mask_dir = join(mask_dir, 'R001_X006_Y008', 'random_gaussian_0', 'merged_fraction_mask')
		else:
			mask_dir = join(mask_dir, 'R001_X004_Y004', 'random_gaussian_0', 'merged_fraction_mask')

		metric_image = []
		for i in [0.0, 0.2, 0.8]:

			if i == 0.0:
				with open(join(os.path.dirname(mask_dir), 'repaired_mask', 'metrics_' + 'cell_matched_mask_deepcell_membrane-0.12.3.pickle_v28.json'), 'r') as f:
					metrics = json.load(f)
			else:
				with open(join(mask_dir, 'metrics_' + 'cell_matched_mask_deepcell_membrane-0.12.3' + '_' + str(i) + '.pickle_v28.json'), 'r') as f:
					metrics = json.load(f)
			test = pd.DataFrame.from_dict(metrics)
			metric_mask = []
			metric_mask.append(np.array(list(metrics['Matched Cell'].items()))[:, 1])
			metric_mask.append(np.array(list(metrics['Nucleus (including nuclear membrane)'].items()))[:, 1])
			metric_mask.append(np.array(list(metrics['Cell Not Including Nucleus (cell membrane plus cytoplasm)'].items()))[:, 1])
			metric_mask = np.hstack(metric_mask)
			metric_mask = metric_mask.astype(np.float64)
			metric_image.append(metric_mask)
		metric_image =np.stack(metric_image)
		metric_all.append(metric_image)
	metric_all = np.stack(metric_all)
	metric_all = np.mean(metric_all, axis=0)

	fig, ax = plt.subplots()
	width = 0.2
	x = np.arange(len(metric_abre))
	
	ax.bar(x, metric_all[0], width, color='#A5D8FC', label='Original')
	ax.bar(x + width, metric_all[1], width, color='#6593F5', label='90% of original cell num')
	ax.bar(x + (2 * width), metric_all[2], width, color='#000080', label='60% of original cell num')
	# ax.bar(x + (3 * width), metric_all[3], width, color='#0F52BA', label='75% chance merged')
	# ax.bar(x + (4 * width), metric_all[4], width, color='#000080', label='100% chance merged')
	
	ax.set_ylabel('Metric Value')
	ax.set_xlabel('Metric')
	ax.set_ylim(0,1.3)
	ax.set_xticks(x + width + width/2)
	ax.set_xticklabels(metric_abre, rotation = 45, rotation_mode='anchor', ha='right', fontsize=9)
	ax.legend()
	fig.savefig('/Users/hrchen/Downloads/segmentation_RRA/figures/merged_metrics.png', dpi=500, bbox_inches='tight')
	plt.clf()
	# plt.show()
	# plt.close()

	# for i in range(metric_all.shape[0]):
	# 	plt.bar(metric_abre + i, metric_all[i])
	# plt.show()
	#
	# 		quality_score_list_mask.append(quality_score)
	# 	quality_score_list_pieces.append(quality_score_list_mask)
	# merged_quality_score_list = np.stack(quality_score_list_pieces)
	# avg_merged_quality_score_list = np.average(merged_quality_score_list, axis=0)