import json
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import os

if __name__ == '__main__':

	file_dir = os.getcwd()
	data_dir = join(file_dir, 'data', 'intermediate', 'manuscript_v28_underseg', 'CODEX')

	shift_list = ['shifted 0.1%', 'shifted 1%', 'shifted 50%']
	mask_dir_list = sorted(glob.glob(join(data_dir, 'HBM**'), recursive=True))
	quality_score_list_pieces = []
	for mask_dir in mask_dir_list:
		if mask_dir == join(data_dir, 'HBM988.SNDW.698'):
			mask_dir = join(mask_dir, 'R003_X004_Y004', 'random_gaussian_0', 'shifted_mask')
		elif mask_dir == join(data_dir, 'HBM433.MQRQ.278'):
			mask_dir = join(mask_dir, 'R001_X006_Y008', 'random_gaussian_0', 'shifted_mask')
		else:
			mask_dir = join(mask_dir, 'R001_X004_Y004', 'random_gaussian_0', 'shifted_mask')

		quality_score_list_mask = []
		for i in [0.001, 0.01, 0.5]:
			with open(join(mask_dir, 'metrics_' + 'cell_matched_mask_deepcell_membrane-0.12.3' + '_' + str(i) + '.pickle_v28.json'), 'r') as f:
				metrics = json.load(f)
			quality_score = metrics['QualityScore']
			# print(quality_score)

			quality_score_list_mask.append(quality_score)
		quality_score_list_pieces.append(quality_score_list_mask)
	shifted_quality_score_list = np.stack(quality_score_list_pieces)
	avg_shifted_quality_score_list_quality_score_list = np.average(shifted_quality_score_list, axis=0)

	# merged_list = ['DeepCell 0.12.3 mem 95% after merged', 'DeepCell 0.12.3 mem 90% after merged', 'DeepCell 0.12.3 mem 75% after merged', 'DeepCell 0.12.3 mem 60% after merged']
	merged_list = ['90% original cell num', '60% original cell num']
	mask_dir_list = sorted(glob.glob(join(data_dir, 'HBM**'), recursive=True))

	quality_score_list_pieces = []
	for mask_dir in mask_dir_list:
		if mask_dir == join(data_dir, 'HBM988.SNDW.698'):
			mask_dir = join(mask_dir, 'R003_X004_Y004', 'random_gaussian_0', 'merged_fraction_mask')
		elif mask_dir == join(data_dir, 'HBM433.MQRQ.278'):
			mask_dir = join(mask_dir, 'R001_X006_Y008', 'random_gaussian_0', 'merged_fraction_mask')
		else:
			mask_dir = join(mask_dir, 'R001_X004_Y004', 'random_gaussian_0', 'merged_fraction_mask')

		quality_score_list_mask = []
		for i in [0.2, 0.8]:
			with open(join(mask_dir, 'metrics_' + 'cell_matched_mask_deepcell_membrane-0.12.3' + '_' + str(i) + '.pickle_v28.json'), 'r') as f:
				metrics = json.load(f)
			quality_score = metrics['QualityScore']
			# print(quality_score)

			quality_score_list_mask.append(quality_score)
		quality_score_list_pieces.append(quality_score_list_mask)
	merged_quality_score_list = np.stack(quality_score_list_pieces)
	avg_merged_quality_score_list = np.average(merged_quality_score_list, axis=0)



	method_list = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	           'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new',
	           'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']

	methods_abre = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto',
	                            'DeepCell 0.9.0 mem',
	                            'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                            'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
	                            'AICS(classic)',
	                            'Cellsegm', 'Voronoi']

	mask_dir_list = sorted(glob.glob(join(data_dir, 'HBM**'), recursive=True))
	quality_score_list_pieces = []
	for mask_dir in mask_dir_list:
		if mask_dir == join(data_dir, 'HBM988.SNDW.698'):
			mask_dir = join(mask_dir, 'R003_X004_Y004', 'random_gaussian_0', 'repaired_mask')
		elif mask_dir == join(data_dir, 'HBM433.MQRQ.278'):
			mask_dir = join(mask_dir, 'R001_X006_Y008', 'random_gaussian_0', 'repaired_mask')
		else:
			mask_dir = join(mask_dir, 'R001_X004_Y004', 'random_gaussian_0', 'repaired_mask')

		quality_score_list_mask = []
		for method in method_list:
			try:
				with open(join(mask_dir, 'metrics_cell_matched_mask_' + method + '.pickle_v28.json'), 'r') as f:
					metrics = json.load(f)
				quality_score = metrics['QualityScore']

			except:
				quality_score = 0

			quality_score_list_mask.append(quality_score)
		quality_score_list_pieces.append(quality_score_list_mask)
	methods_quality_score_list = np.stack(quality_score_list_pieces)
	avg_methods_quality_score_list = np.average(methods_quality_score_list, axis=0)

	quality_score_list = np.hstack((avg_methods_quality_score_list, avg_merged_quality_score_list))
	quality_score_list = np.hstack((quality_score_list, avg_shifted_quality_score_list_quality_score_list))
	quality_score_name = methods_abre + merged_list + shift_list

	df = pd.DataFrame(
		dict(
			quality_score=quality_score_list,
			methods=quality_score_name,
		)
	)

	df_sorted = df.sort_values('quality_score', ascending=False)
	
	values = np.array([2, 5, 3, 6, 4, 7, 1])
	idx = np.array(list('abcdefg'))
	methods = df_sorted['methods']
	clrs = []
	for method in methods:
		if 'shifted' in method:
			clrs.append('green')
		elif 'original' in method:
			clrs.append('red')
		else:
			clrs.append('blue')
	
	sns.set_color_codes("pastel")
	sns.barplot(x="quality_score", y="methods", data=df_sorted, palette=clrs)
	
	plt.xlabel('Quality Score')
	plt.ylabel('Method')
	plt.savefig(join(file_dir, 'figures', 'original_vs_merged_vs_shifted.png'), dpi=500, bbox_inches='tight')
	plt.clf()
