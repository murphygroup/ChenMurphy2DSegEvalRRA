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
import cv2

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_neighboring_cells(mask, coord, cell_index):
	x_max = np.max(coord[0])
	x_min = np.min(coord[0])
	y_max = np.max(coord[1])
	y_min = np.min(coord[1])
	cropped_cell_mask = mask[max(x_min-5, 0):min(x_max+6, mask.shape[0]), max(y_min-5, 0):min(y_max+6, mask.shape[1])]
	cropped_cell_mask_single_cell = cropped_cell_mask.copy()
	cropped_cell_mask_single_cell[cropped_cell_mask_single_cell != cell_index] = 0
	mask_boundary = find_boundaries(cropped_cell_mask_single_cell, mode='outer')
	mask_boundary_indexed = np.unique(get_indexed_mask(cropped_cell_mask, mask_boundary))
	# mask_boundary_indexed = np.unique(cropped_cell_mask)
	mask_boundary_indexed = np.delete(mask_boundary_indexed, np.where(mask_boundary_indexed == 0))
	mask_boundary_indexed = np.delete(mask_boundary_indexed, np.where(mask_boundary_indexed == cell_index))
	return mask_boundary_indexed


def cut_cell(x_coord, y_coord):
	if len(np.unique(x_coord)) > len(np.unique(y_coord)):
		x_cut = np.round(np.average(x_coord)).astype(int)
		cell_1_coord = (x_coord[x_coord < x_cut], y_coord[x_coord < x_cut])
		cell_1_boundary_coord = (x_coord[x_coord == (x_cut-1)], y_coord[x_coord == (x_cut-1)])
		cell_2_coord = (x_coord[x_coord >= x_cut], y_coord[x_coord >= x_cut])
		cell_2_boundary_coord = (x_coord[x_coord == x_cut], y_coord[x_coord == x_cut])
	else:
		y_cut = np.round(np.average(y_coord)).astype(int)
		cell_1_coord = (x_coord[y_coord < y_cut], y_coord[y_coord < y_cut])
		cell_1_boundary_coord = (x_coord[y_coord == (y_cut-1)], y_coord[y_coord == (y_cut-1)])
		cell_2_coord = (x_coord[y_coord >= y_cut], y_coord[y_coord >= y_cut])
		cell_2_boundary_coord = (x_coord[y_coord == y_cut], y_coord[y_coord == y_cut])
		
	return cell_1_coord, cell_2_coord, cell_1_boundary_coord, cell_2_boundary_coord

def get_mask_new_indices(mask):
	new_mask = np.zeros(mask.shape)
	cell_index = 1
	cell_coords_new = get_indices_sparse(mask)[1:]
	for i in range(len(cell_coords_new)):
		if len(cell_coords_new[i][0]) != 0:
			new_mask[cell_coords_new[i]] = cell_index
			cell_index += 1
	return new_mask.astype(np.int64)


def splitter(mask, nucleus, cell_outside_nucleus):
	max_mask_index = np.max(mask)
	cell_coords = get_indices_sparse(mask)[1:]
	# for i in range(len(cell_coords)):
	for i in random.sample(range(0, len(cell_coords)), 1):
		if len(cell_coords[i][0]) != 0:
			x_coord = cell_coords[i][0]
			y_coord = cell_coords[i][1]
			cut_cell_1_coord, cut_cell_2_coord, boundary_cell_1_coord, boundary_cell_2_coord = cut_cell(x_coord, y_coord)
			if (sum(np.sign(nucleus[cut_cell_2_coord])) != sum(np.sign(nucleus[cell_coords[i]]))) and (sum(np.sign(nucleus[cut_cell_1_coord])) != sum(np.sign(nucleus[cell_coords[i]]))):
				max_mask_index += 1
				mask[cut_cell_2_coord] = max_mask_index
				nucleus[cut_cell_2_coord] = np.sign(nucleus[cut_cell_2_coord]) * max_mask_index
				nucleus[boundary_cell_1_coord] = 0
				nucleus[boundary_cell_2_coord] = 0
				cell_outside_nucleus[cut_cell_2_coord] = np.sign(cell_outside_nucleus[cut_cell_2_coord]) * max_mask_index
				cell_outside_nucleus[boundary_cell_1_coord] = i
				cell_outside_nucleus[boundary_cell_2_coord] = np.sign(cell_outside_nucleus[boundary_cell_2_coord]) * max_mask_index
	return mask, nucleus, cell_outside_nucleus


def merger(mask, nucleus, cell_outside_nucleus, percentage):
	cell_coords = get_indices_sparse(mask)[1:]
	nucleus_coords = get_indices_sparse(nucleus)[1:]
	cell_outside_nucleus_coords = get_indices_sparse(cell_outside_nucleus)[1:]
	merged_cell_list = []
	merged_cell_list_to = []
	cell_num = len(cell_coords)
	# for i in random.sample(range(0, len(cell_coords)), round(len(cell_coords) * percentage)):
	cell_num_list = np.linspace(0, cell_num-1, cell_num, dtype=int)
	# np.random.shuffle(cell_num_list)
	for i in range(cell_num):
		if len(cell_coords[i][0]) != 0 and (i+1) not in merged_cell_list:
			cell_coord = cell_coords[i]
			neighboring_cell_indices = get_neighboring_cells(mask, cell_coord, i+1)
			# neighboring_cell_indices = np.delete(neighboring_cell_indices, np.where(neighboring_cell_indices == 0))
			# neighboring_cell_indices = np.delete(neighboring_cell_indices, np.where(neighboring_cell_indices == (i+1)))
			if len(neighboring_cell_indices) != 0:
				for j in neighboring_cell_indices:
					# merge_index = np.random.choice(np.arange(0, 2), p=[1-percentage, percentage])
					# if j not in merged_cell_list and merge_index == 1:
					if j not in merged_cell_list:
					# print(j)
						mask[cell_coords[j-1]] = i+1
						nucleus[nucleus_coords[j-1]] = i+1
						cell_outside_nucleus[cell_outside_nucleus_coords[j-1]] = i+1
						merged_cell_list.append(j)
						merged_cell_list.append(i+1)
						merged_cell_list_to.append(i+1)
						break
	# mask = get_mask_new_indices(mask)
	# nucleus = get_mask_new_indices(nucleus)
	# cell_outside_nucleus = get_mask_new_indices(cell_outside_nucleus)
				print(len(merged_cell_list) / cell_num)
				if len(merged_cell_list) / cell_num > percentage:
					return mask, nucleus, cell_outside_nucleus, merged_cell_list_to

def merge_nuclei(nuclear_mask, cell_outside_nucleus_mask, cell_list):
	nucleus_coords_all = get_indices_sparse(nuclear_mask)[1:]
	cell_outside_nucleus_coords_all = get_indices_sparse(cell_outside_nucleus_mask)[1:]
	for cell_index in cell_list:
		if len(nucleus_coords_all[cell_index-1][0]) != 0:
			nucleus_coord = nucleus_coords_all[cell_index-1]
			cell_outside_nucleus_coord = cell_outside_nucleus_coords_all[cell_index-1]
			x_max = np.max(nucleus_coord[0])
			x_min = np.min(nucleus_coord[0])
			y_max = np.max(nucleus_coord[1])
			y_min = np.min(nucleus_coord[1])
			single_nucleus_mask = nuclear_mask[max(x_min-10, 0):min(x_max+11, nuclear_mask.shape[0]), max(y_min-10, 0):min(y_max+11, nuclear_mask.shape[1])]
			single_nucleus_mask_original = single_nucleus_mask.copy()
			single_nucleus_mask[single_nucleus_mask != cell_index] = 0
			single_nucleus_mask_inverted = -np.sign(single_nucleus_mask)+1
			single_nucleus_mask_inverted_opened = binary_opening(single_nucleus_mask_inverted, selem=disk(3)) * 1
			single_nucleus_mask_opened = -single_nucleus_mask_inverted_opened + 1
			
			single_nucleus_mask_final = np.sign(single_nucleus_mask + single_nucleus_mask_opened) * cell_index
			single_nucleus_mask_original[np.where(single_nucleus_mask_final == cell_index)] = cell_index
			nuclear_mask[max(x_min-10, 0):min(x_max+11, nuclear_mask.shape[0]), max(y_min-10, 0):min(y_max+11, nuclear_mask.shape[1])] = single_nucleus_mask_original
			
			single_nucleus_mask = nuclear_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])]
			single_cell_outside_nucleus_mask = cell_outside_nucleus_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])]
			single_cell_outside_nucleus_mask[np.where(single_nucleus_mask != 0)] = 0

			# single_cell_mask = cell_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])]
			# single_cell_mask[single_cell_mask != cell_index] = 0
			# single_cell_mask_boundaries = find_boundaries(single_cell_mask, mode='inner') * 1
			# single_nucleus_mask[single_cell_mask_boundaries == 1] = 0
			# single_cell_outside_nucleus_mask[single_cell_mask_boundaries == 1] = cell_index
			
			# nuclear_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])] = single_nucleus_mask
			cell_outside_nucleus_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])] = single_cell_outside_nucleus_mask

			
			# plt.imshow(single_cell_outside_nucleus_mask)
			# plt.show()
			# plt.clf()
			# single_nucleus_mask_closed = binary_closing(single_nucleus_mask) * 1
			
			# nuclear_mask[max(x_min-2, 0):min(x_max+2, nuclear_mask.shape[0]), max(y_min-2, 0):min(y_max+2, nuclear_mask.shape[1])] = single_nucleus_mask_closed * cell_index
			# single_cell_outside_nucleus_mask = cell_outside_nucleus_mask[max(x_min-2, 0):min(x_max+2, nuclear_mask.shape[0]), max(y_min-2, 0):min(y_max+2, nuclear_mask.shape[1])]
			# cell_outside_nucleus_mask[max(x_min-2, 0):min(x_max+2, nuclear_mask.shape[0]), max(y_min-2, 0):min(y_max+2, nuclear_mask.shape[1])] = single_cell_outside_nucleus_mask * single_nucleus_mask_closed
	return nuclear_mask, cell_outside_nucleus_mask
if __name__ == '__main__':
	np.random.seed(3)
	# data_dir = '/data/HuBMAP/CODEX/HBM244.TJLK.223/R001_X004_Y004/random_gaussian_0'
	dataset_list = sorted(glob.glob('/data/HuBMAP/CODEX/HBM**'), reverse=True)
	# dataset_list = ['/data/HuBMAP/CODEX/HBM988.SNDW.698']
	
	for dataset in dataset_list:
		if True:
			print(dataset)
			if dataset == '/data/HuBMAP/CODEX/HBM988.SNDW.698':
				data_dir = join(dataset, 'R003_X004_Y004', 'random_gaussian_0')
			elif dataset == '/data/HuBMAP/CODEX/HBM433.MQRQ.278':
				data_dir = join(dataset, 'R001_X006_Y008', 'random_gaussian_0')
			else:
				data_dir = join(dataset, 'R001_X004_Y004', 'random_gaussian_0')
		# if True:
			cell_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'cell_matched_mask_deepcell_membrane-0.12.3.pickle'), 'r')).astype(np.int64)
			nuclear_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'nuclear_matched_mask_deepcell_membrane-0.12.3.pickle'), 'r')).astype(np.int64)
			cell_outside_nucleus_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'cell_outside_nucleus_matched_mask_deepcell_membrane-0.12.3.pickle'), 'r')).astype(np.int64)
			# split_cell_mask, split_nuclear_mask, split_cell_outside_nucleus_mask = splitter(cell_mask, nuclear_mask, cell_outside_nucleus_mask)
			# pickle.dump(split_cell_mask, bz2.BZ2File(join(data_dir, 'split_mask', 'cell_matched_mask_deepcell_membrane-0.12.3.pickle'), 'w'))
			# pickle.dump(split_nuclear_mask, bz2.BZ2File(join(data_dir, 'split_mask', 'nuclear_matched_mask_deepcell_membrane-0.12.3.pickle'), 'w'))
			# pickle.dump(split_cell_outside_nucleus_mask, bz2.BZ2File(join(data_dir, 'split_mask', 'cell_outside_nucleus_matched_mask_deepcell_membrane-0.12.3.pickle'), 'w'))
			# os.system('rm ' + join(data_dir, 'merged_mask', '*pickle'))
			# os.system('rm ' + join(data_dir, 'merged_fraction_mask', '*json'))
			# for i in np.linspace(0.5,1,2):
			for i in [0.1, 0.2, 0.5, 0.8]:
				print(i)
				if True:
					merged_cell_mask, merged_nuclear_mask, merged_cell_outside_nucleus_mask, merged_cell_list = merger(cell_mask.copy(), nuclear_mask.copy(), cell_outside_nucleus_mask.copy(), i)
					# nuclear_mask = merged_nuclear_mask.copy()
					# cell_outside_nucleus_mask = merged_cell_outside_nucleus_mask.copy()
					# cell_list = merged_cell_list
					merged_nuclear_mask, merged_cell_outside_nucleus_mask = merge_nuclei(merged_nuclear_mask.copy(), merged_cell_outside_nucleus_mask.copy(), merged_cell_list)
					if not os.path.exists(join(data_dir, 'merged_fraction_mask')):
						os.makedirs(join(data_dir, 'merged_fraction_mask'))
					
					merged_cell_mask = get_mask_new_indices(merged_cell_mask)
					merged_nuclear_mask = get_mask_new_indices(merged_nuclear_mask)
					merged_cell_outside_nucleus_mask = get_mask_new_indices(merged_cell_outside_nucleus_mask)
					print(len(np.unique(merged_cell_mask)))
					print(join(data_dir, 'merged_fraction_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_' + str(i) + '.pickle'))
					
					pickle.dump(merged_cell_mask, bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_' + str(i) + '.pickle'), 'w'))
					pickle.dump(merged_nuclear_mask, bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'nuclear_matched_mask_deepcell_membrane-0.12.3_' + str(i) + '.pickle'), 'w'))
					pickle.dump(merged_cell_outside_nucleus_mask, bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'cell_outside_nucleus_matched_mask_deepcell_membrane-0.12.3_' + str(i) + '.pickle'), 'w'))
				
		
		'''
		# if True:
		cell_mask = pickle.load(bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_1.0.pickle'), 'r')).astype(np.int64)
		nuclear_mask = pickle.load(bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_1.0.pickle'), 'r')).astype(np.int64)
		cell_outside_nucleus_mask = pickle.load(bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_1.0.pickle'), 'r')).astype(np.int64)
		# split_cell_mask, split_nuclear_mask, split_cell_outside_nucleus_mask = splitter(cell_mask, nuclear_mask, cell_outside_nucleus_mask)
		# pickle.dump(split_cell_mask, bz2.BZ2File(join(data_dir, 'split_mask', 'cell_matched_mask_deepcell_membrane-0.12.3.pickle'), 'w'))
		# pickle.dump(split_nuclear_mask, bz2.BZ2File(join(data_dir, 'split_mask', 'nuclear_matched_mask_deepcell_membrane-0.12.3.pickle'), 'w'))
		# pickle.dump(split_cell_outside_nucleus_mask, bz2.BZ2File(join(data_dir, 'split_mask', 'cell_outside_nucleus_matched_mask_deepcell_membrane-0.12.3.pickle'), 'w'))
		# os.system('rm ' + join(data_dir, 'merged_fraction_mask', '*pickle'))
		# os.system('rm ' + join(data_dir, 'merged_fraction_mask', '*json'))
		for i in [1]:
			# for i in [1]:
			print(2.0)
			merged_cell_mask, merged_nuclear_mask, merged_cell_outside_nucleus_mask, merged_cell_list = merger(cell_mask.copy(), nuclear_mask.copy(), cell_outside_nucleus_mask.copy(), i)
			# nuclear_mask = merged_nuclear_mask.copy()
			# cell_outside_nucleus_mask = merged_cell_outside_nucleus_mask.copy()
			# cell_list = merged_cell_list
			# merged_nuclear_mask_new, merged_cell_outside_nucleus_mask = merge_nuclei(merged_nuclear_mask.copy(), merged_cell_outside_nucleus_mask.copy(), merged_cell_list)
			if not os.path.exists(join(data_dir, 'merged_fraction_mask')):
				os.makedirs(join(data_dir, 'merged_fraction_mask'))
			print(len(np.unique(merged_cell_mask)))
			print(join(data_dir, 'merged_fraction_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_' + '2.0' + '.pickle'))
			pickle.dump(merged_cell_mask, bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'cell_matched_mask_deepcell_membrane-0.12.3_' + '2.0' + '.pickle'), 'w'))
			pickle.dump(merged_nuclear_mask, bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'nuclear_matched_mask_deepcell_membrane-0.12.3_' + '2.0' + '.pickle'), 'w'))
			pickle.dump(merged_cell_outside_nucleus_mask, bz2.BZ2File(join(data_dir, 'merged_fraction_mask', 'cell_outside_nucleus_matched_mask_deepcell_membrane-0.12.3_' + '2.0' + '.pickle'), 'w'))
	
		# merged_cell_mask = merger(cell_mask, nuclear_mask, cell_outside_nucleus_mask)
		# plt.imshow(split_cell_outside_nucleus_mask)
		# plt.show()
		# plt.clf()'''
		
		
	# data_list = glob.glob('/data/HuBMAP/CODEX/HBM433.MQRQ.278/**/random_gaussian_0/repaired_mask/cell_matched_mask_deepcell_membrane-0.12.3.pickle', recursive=True)
	# for data_dir in data_list:
	# 	test = pickle.load(bz2.BZ2File(data_dir, 'r'))
	# 	print(data_dir, np.max(test))