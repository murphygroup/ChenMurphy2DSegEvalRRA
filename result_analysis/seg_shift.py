import numpy as np
from os.path import join
import pickle
import bz2
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import random
import os
import glob
from skimage.morphology import binary_closing, binary_opening, disk

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]


def get_mask_new_indices(mask):
	new_mask = np.zeros(mask.shape)
	cell_index = 1
	cell_coords_new = get_indices_sparse(mask)[1:]
	for i in range(len(cell_coords_new)):
		if len(cell_coords_new[i][0]) != 0:
			new_mask[cell_coords_new[i]] = cell_index
			cell_index += 1
	return new_mask.astype(np.int64)

def segmentation_shift(mask, shift_num):
	coords = get_indices_sparse(mask)[1:]
	mask_shape = mask.shape
	for i in range(len(coords)):
		coord = coords[i]
		coord = [coord[0] + shift_num, coord[1] + shift_num]
		coord_0 = coord[0]
		coord_1 = coord[1]
		# coord_0[coord_0 >= mask_shape[0]] = coord_0[coord_0 >= mask_shape[0]] - mask_shape[0]
		# coord_1[coord_1 >= mask_shape[1]] = coord_1[coord_1 >= mask_shape[1]] - mask_shape[1]
		coord_0[coord_0 >= mask_shape[0]] = 0
		coord_1[coord_0 >= mask_shape[0]] = 0
		coord_0[coord_1 >= mask_shape[1]] = 0
		coord_1[coord_1 >= mask_shape[1]] = 0

		coord = [coord_0, coord_1]
		coords[i] = coord
	
	new_mask = np.zeros(mask.shape)
	# cell_index = 1
	for i in range(len(coords)):
		# if np.sum(coords[i][0]) != 0:
		new_mask[coords[i]] = i+1
			# cell_index += 1
		
	return new_mask



def get_final_masks(cell_mask, nuclear_mask, cell_outside_nucleus_mask):
	cell_mask_coords = get_indices_sparse(cell_mask.astype(np.int64))[1:]
	nuclear_mask_coords = get_indices_sparse(nuclear_mask.astype(np.int64))[1:]
	cell_outside_nucleus_mask_coords = get_indices_sparse(cell_outside_nucleus_mask.astype(np.int64))[1:]
	new_cell_mask = np.zeros(cell_mask.shape)
	new_nuclear_mask = np.zeros(nuclear_mask.shape)
	new_cell_outside_nucleus_mask = np.zeros(cell_outside_nucleus_mask.shape)
	cell_index = 1
	for i in range(len(nuclear_mask_coords)):
		if np.sum(nuclear_mask_coords[i][0]) != 0:
			new_cell_mask[cell_mask_coords[i]] = cell_index
			new_nuclear_mask[nuclear_mask_coords[i]] = cell_index
			new_cell_outside_nucleus_mask[cell_outside_nucleus_mask_coords[i]] = cell_index
			cell_index += 1
	return new_cell_mask, new_nuclear_mask, new_cell_outside_nucleus_mask
		
	

if __name__ == '__main__':
	np.random.seed(3)
	dataset_list = sorted(
		glob.glob('/Volumes/Extreme/segmentation/CODEX/HBM433.MQRQ.278/R001_X006_Y008/random_gaussian_0'))
	dataset_list = dataset_list + sorted(
		glob.glob('/Volumes/Extreme/segmentation/CODEX/HBM988.SNDW.698//R003_X004_Y004/random_gaussian_0'))
	dataset_list = dataset_list + sorted(
		glob.glob('/Volumes/Extreme/segmentation/CODEX/**/R001_X004_Y004/random_gaussian_0'))
	dataset_list = ['/Volumes/Extreme/segmentation/CODEX/HBM355.JDLK.244/R001_X004_Y004/random_gaussian_0']
	# if True:
	for shift_percentage in [0.01]:
		for data_dir in dataset_list:
			print(data_dir)
			cell_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'cell_matched_mask_deepcell_membrane-0.12.3.pickle'), 'r')).astype(np.int64)
			nuclear_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'nuclear_matched_mask_deepcell_membrane-0.12.3.pickle'), 'r')).astype(np.int64)
			cell_outside_nucleus_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'cell_outside_nucleus_matched_mask_deepcell_membrane-0.12.3.pickle'), 'r')).astype(np.int64)
			# shift_percentage = 0.01
			shift = round(((cell_mask.shape[0] + cell_mask.shape[0]) / 2 * shift_percentage))
			cell_mask_shifted = segmentation_shift(cell_mask, shift)
			nuclear_mask_shifted = segmentation_shift(nuclear_mask, shift)
			cell_outside_nucleus_mask_shifted = segmentation_shift(cell_outside_nucleus_mask, shift)
			cell_mask_final, nuclear_mask_final, cell_outside_nucleus_mask_final = get_final_masks(cell_mask_shifted, nuclear_mask_shifted, cell_outside_nucleus_mask_shifted)
			# np.max(cell_mask_shifted)
			# np.max(cell_outside_nucleus_mask_shifted)
			# plt.imshow(cell_outside_nucleus_mask_final)
			# plt.show()
		
			if not os.path.exists(join(data_dir, 'shifted_mask')):
				os.makedirs(join(data_dir, 'shifted_mask'))
			print(join(data_dir, 'shifted_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_'+ str(shift_percentage) + '.pickle'))
			pickle.dump(cell_mask_final, bz2.BZ2File(join(data_dir, 'shifted_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_'+ str(shift_percentage) + '.pickle'), 'w'))
			pickle.dump(nuclear_mask_final, bz2.BZ2File(join(data_dir, 'shifted_mask', 'nuclear_matched_mask_deepcell_membrane-0.12.3_'+ str(shift_percentage) + '.pickle'), 'w'))
			pickle.dump(cell_outside_nucleus_mask_final, bz2.BZ2File(join(data_dir, 'shifted_mask', 'cell_outside_nucleus_matched_mask_deepcell_membrane-0.12.3_'+ str(shift_percentage) + '.pickle'), 'w'))
		