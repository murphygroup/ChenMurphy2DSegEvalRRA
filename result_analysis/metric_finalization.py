import bz2
import numpy as np
import os
import sys
from os.path import join
import glob
from skimage.io import imread
import pickle

def flatten(t):
	return [item for sublist in t for item in sublist]

def get_concatenated_compartments_data(input_dir):
	
	compartment_list = ['nuclear_matched', 'cell_outside_nucleus_matched']
	features_3D_whole_cell = pickle.load(bz2.BZ2File(join(input_dir, 'cell_matched', 'feature_3D.pickle'), 'r'))
	features_3D = features_3D_whole_cell[:, :, :-3]
	for c in compartment_list:
		fig_dir = join(input_dir, c)
		features_3D_c = pickle.load(bz2.BZ2File(join(fig_dir, 'feature_3D.pickle'), 'r'))
		# only need last metrics of cell homogeneity from cell compartments
		features_3D = np.dstack((features_3D, features_3D_c[:, :, -3:]))
	compartment_dir = join(input_dir, compartment)
	if not os.path.exists(compartment_dir):
		os.makedirs(compartment_dir)

	pickle.dump(features_3D, bz2.BZ2File(join(compartment_dir, 'feature_3D.pickle'), 'w'))
	
	img_names = pickle.load(bz2.BZ2File(join(input_dir, 'cell_matched', 'img_names.pickle'), 'r'))
	pickle.dump(img_names, bz2.BZ2File(join(compartment_dir, 'img_names.pickle'), 'w'))
	
	return features_3D

	
def feature_extraction(dir, method, n):
	
	cell_basic_method = np.empty((1, 6))
	# img_shape = np.load(join(os.path.dirname(os.path.dirname(dir)), 'random_gaussian_0', 'img_shape.npy'))
	img_shape = np.load(join(os.path.dirname(dir), 'img_shape.npy'))

	if noise_type == 'downsampling' and n != 0:
		pixel_factor = (n / 100) ** 2
	else:
		pixel_factor = 1
	micron_num = pixel_size / pixel_factor * img_shape[0] * img_shape[1]
	
	# if (noise_type == 'downsampling' and n != 0) or (method == 'aics_classic'):
	# 	metric_name = 'resize_metric'
	# else:
	
	metric_name = 'metric'
	 
	try:
		cell_basic_method[0, :] = np.loadtxt(join(dir, metric_name + '_cell_basic_' + method + '.txt'))
		if cell_basic_method[0, 0] <= 1 or cell_basic_method[0, 5] == 0.0:
			cell_basic_method = np.expand_dims(np.array((0, 0, 0, 0, 0)), axis=0)
		else:
			cell_basic_method = np.delete(cell_basic_method, 1, axis=1)
			cell_basic_method[0, 0] = cell_basic_method[0, 0] / micron_num * 100
			cell_basic_method[0, 2] = 1 - cell_basic_method[0, 2]
			cell_basic_method[0, 4] = 1 / (np.log(cell_basic_method[0, 4]) + 1)
		
	except:
		cell_basic_method = np.expand_dims(np.array((0, 0, 0, 0, 0)), axis=0)



	foreground_method = np.empty((1, 2))
	try:
		foreground_method[0, :] = np.loadtxt(join(dir, metric_name + '_foreground_' + method + '.txt'))
		foreground_method[0, 0] = 1 / (foreground_method[0, 0] + 1)
	except:
		foreground_method = np.expand_dims(np.array((0, 0)), axis=0)
	
	try:
		cell_current = np.loadtxt(join(dir, metric_name + '_cell_' + method + '.txt'))
		cell_current_num = cell_current.shape[1]
		cell_CV = np.empty((0, cell_current_num))
		cell_PC = np.empty((0, cell_current_num))
		cell_sil = np.empty((0, cell_current_num))
		cell_CV = np.vstack((cell_CV, cell_current[0, :]))
		cell_PC = np.vstack((cell_PC, cell_current[1, :]))
		cell_sil = np.vstack((cell_sil, cell_current[2, :]))
		cell_method = np.empty((1, 3))
		cell_sil[0, 0] = 0
		if np.average(cell_CV) == 0:
			avg_CV = 0
		else:
			avg_CV = 1 / (np.average(cell_CV) + 1)
		cell_method[0] = [avg_CV, np.average(cell_PC), np.average(cell_sil[0][1:])]
	except:
		cell_method = np.expand_dims(np.array((0,0,-1)), axis=0)

	compartment_matching = np.empty((1, 1))
	if sys.argv[2] == 'cell_matched':
		try:
			if os.path.exists(join(dir, 'fraction_matched_cell_nuclei_num_' + method + '.txt')):
				compartment_matching_cell_nuclear = np.loadtxt(join(dir, 'fraction_matched_cell_nuclei_num_' + method + '.txt'))
			else:
				compartment_matching_cell_nuclear = np.loadtxt(join(dir, metric_name + '_fraction_matched_cell_nuclei_num_' + method + '.txt'))
				
			compartment_matching[0, 0] = compartment_matching_cell_nuclear
		except:
			compartment_matching[0, 0] = 0
	
	if sys.argv[2] == 'cell_matched':
		feature = np.hstack((cell_basic_method, compartment_matching, foreground_method, cell_method))
	else:
		feature = np.hstack((cell_basic_method, foreground_method, cell_method))

	if feature[0, 0] == 0:
		feature[0, :] = 0
		feature[0, -1] = -1
		
	return feature



def feature_matrix(n):
	feature_all = np.empty((0, method_num, metric_num))
	modality = os.path.basename(data_dir)
	if tissue == 'all_tissues':
		if n == 0:
			result_dir_list = sorted(glob.glob(data_dir + '/**/random_gaussian_0', recursive=True))
		elif noise_type == 'gaussian':
			result_dir_list = sorted(glob.glob(data_dir + '/**/random_gaussian_' + str(n), recursive=True))
		else:
			result_dir_list = sorted(glob.glob(data_dir + '/**/downsampling_' + str(n), recursive=True))
	else:
		result_dir_list = []
		for image_idx in range(len(checklist)):
			if n == 0:
				result_dir_list = result_dir_list + sorted(glob.glob(data_dir + '/' + checklist[image_idx] + '/**/random_gaussian_0', recursive=True))
			elif noise_type == 'gaussian':
				result_dir_list = result_dir_list + sorted(glob.glob(data_dir + '/' + checklist[image_idx] + '/**/random_gaussian_' + str(n), recursive=True))
			else:
				result_dir_list = result_dir_list + sorted(glob.glob(data_dir + '/' + checklist[image_idx] + '/**/downsampling_' + str(n), recursive=True))
	for result_dir in result_dir_list:
		feature_tile = np.empty((0, metric_num))
		for method in methods:
			feature_method = feature_extraction(join(result_dir, 'result_' + repaired + '_' + compartment), method, n)
			feature_tile = np.vstack((feature_tile, feature_method))
		feature_all = np.vstack((feature_all, feature_tile[np.newaxis, ...]))
	result_dir_list_output = []
	for single_result_dir in result_dir_list:
		result_dir_list_output.append(single_result_dir[single_result_dir.index(modality):])
	return feature_all, result_dir_list_output


if __name__ == '__main__':
	file_dir = os.getcwd()
	script_dir = join(file_dir, 'scripts')
	save_dir = join(file_dir, 'data')
	for noise_type in ['gaussian', 'downsampling']:
		data_dir = sys.argv[1]
		data_type = data_dir.split('/')[-1]
		compartment = sys.argv[2]
		tissue = sys.argv[3]
		repaired = sys.argv[4]
		if not os.path.exists(join(save_dir, 'metrics')):
			os.makedirs(join(save_dir, 'metrics'))
		tissue_dir = join(save_dir, 'metrics', noise_type, repaired, data_type, tissue, sys.argv[2])
		if not os.path.exists(tissue_dir):
			os.makedirs(tissue_dir)
			
		# get tissue info
		if data_type == 'CODEX':
			checklist = np.loadtxt(join(script_dir, 'CODEX_img_list.txt'), dtype=str)
			if not tissue == 'all_tissues':
				checklist = checklist[np.where(checklist[:, 3] == tissue), 0]
				checklist = np.squeeze(checklist, axis=0)
				img_num = len(checklist)
		if data_type == 'IMC':
			checklist = np.loadtxt(join(script_dir, 'IMC_img_list.txt'), dtype=str)
			if not tissue == 'all_tissues':
				checklist = checklist[np.where(checklist[:, 2] == tissue), 0]
				checklist = np.squeeze(checklist, axis=0)
				img_num = len(checklist)
		
		# get pixel size of modality
		if data_type == 'CODEX':
			pixel_size = 0.37745 ** 2
		elif data_type == 'CellDIVE':
			pixel_size = 0.325 ** 2
		elif data_type == 'MIBI':
			pixel_size = 0.391 ** 2
		elif data_type == 'IMC':
			pixel_size = 1 ** 2
		noise_num = 3
		noise_interval = 1
		feature_types = ['cell_basic', 'foreground', 'background', 'cell']
	

		methods = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new', 'deepcell_cytoplasm_new', 'deepcell_membrane',  'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new', 'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
		methods_abre = ['DeepCell 0.12.3 cell membrane', 'DeepCell 0.12.3 cytoplasm', 'DeepCell 0.9.0 cell membrane', 'DeepCell 0.9.0 cytoplasm', 'DeepCell 0.6.0 cell membrane', 'DeepCell 0.6.0 cytoplasm', 'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX', 'AICS(classic)',  'Cellsegm', 'Voronoi']

		method_num = len(methods)
		if sys.argv[2] == 'cell_matched':
			metric_num = 11
		else:
			metric_num = 10
		
		# collecting features
		if noise_type == 'gaussian':
			noise_range = np.arange(0, noise_interval * (noise_num + 1), noise_interval)
		else:
			noise_range = [0, 70, 50, 30]
		noise_level = ['Low', 'Medium', 'High']
		total_tile_num = []
		img_names_all = []
		if compartment == 'concatenated_compartments':
			feature_all_3D_final = get_concatenated_compartments_data(os.path.dirname(tissue_dir))
			total_tile_num = pickle.load(bz2.BZ2File(join(os.path.dirname(tissue_dir), 'cell_matched', 'total_tile_num.pickle'), 'r'))
			
		else:
			for noise in noise_range:
				features, img_names = feature_matrix(noise)
				img_names_all.append(img_names)
				total_tile_num.append(features.shape[0])
				if noise == noise_range[0]:
					feature_all_3D = features
					feature_all_3D_0 = features.copy()
				else:
					feature_all_3D = np.vstack((feature_all_3D, features))
	
			feature_all_3D_final = feature_all_3D
			metric_num = feature_all_3D_final.shape[2]
			feature_all_3D_final_save = feature_all_3D_final
			pickle.dump(feature_all_3D_final_save, bz2.BZ2File(join(tissue_dir, 'feature_3D.pickle'), 'w'))
		print('output ', tissue_dir)
		pickle.dump(total_tile_num, bz2.BZ2File(join(tissue_dir, 'total_tile_num.pickle'), 'w'))
		if compartment != 'concatenated_compartments':
			pickle.dump(flatten(img_names_all), bz2.BZ2File(join(tissue_dir, 'img_names.pickle'), 'w'))

