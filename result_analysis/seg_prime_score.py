import os
import numpy as np
import sys
from os.path import join
import bz2
import pickle
from scipy.sparse import csr_matrix
import pandas as pd


def check_match_overlap(ref_arr, que_arr):
	a = set((tuple(i) for i in ref_arr))
	b = set((tuple(i) for i in que_arr))
	match_pixel_num = len(list(a & b))
	ref_pixel_num = len(list(a))
	if match_pixel_num > (0.5 * ref_pixel_num):
		return True
	else:
		return False
	
def calculate_jaccard(ref_arr, que_arr):
	a = set((tuple(i) for i in ref_arr))
	b = set((tuple(i) for i in que_arr))
	match_pixel_num = len(list(a & b))
	union_pixel_num = len(list(a | b))
	j_idx = match_pixel_num / union_pixel_num
	return j_idx



def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
					  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def get_seg_score(reference_mask, query_mask):
	reference_mask_coords = get_indices_sparse(reference_mask)[1:]
	query_mask_coords = get_indices_sparse(query_mask)[1:]
	reference_mask_coords = list(map(lambda x: np.array(x).T, reference_mask_coords))
	query_mask_coords = list(map(lambda x: np.array(x).T, query_mask_coords))
	
	# reference_mask_matched_index_list = []
	query_mask_matched_index_list = []
	
	seg_score_list = []
	for i in range(len(reference_mask_coords)):
		if len(reference_mask_coords[i]) != 0:
			current_reference_mask_coords = reference_mask_coords[i]
			query_mask_search_num = np.unique(list(map(lambda x: query_mask[tuple(x)], current_reference_mask_coords)))
			best_jaccard = 0
			for j in query_mask_search_num:
				current_query_mask_coords = query_mask_coords[j - 1]
				if j != 0:
					if (j - 1 not in query_mask_matched_index_list):
						match_bool = check_match_overlap(current_reference_mask_coords, current_query_mask_coords)
						if match_bool == True:
							current_jaccard = calculate_jaccard(current_reference_mask_coords,
							                                    current_query_mask_coords)
							if current_jaccard > best_jaccard:
								best_jaccard = current_jaccard
								# i_ind_best = i
								j_ind_best = j - 1
			
			if best_jaccard > 0:
				# reference_mask_matched_index_list.append(i_ind_best)
				query_mask_matched_index_list.append(j_ind_best)
			seg_score_list.append(best_jaccard)
	
	return np.average(seg_score_list)


if __name__ == '__main__':
	
	data_dir = '/home/haoranch/projects/HuBMAP/CODEX_annotated/HBM279.TQRS.775'
	
	tile_list = ['R001_X003_Y004', 'R001_X004_Y003']
	expert_list = ['expert1', 'expert2']
	method_list = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	               'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0',
	               'cellpose_new',
	               'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial', 'expert1', 'expert2']
	
	methods_abre_list = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto', 'DeepCell 0.9.0 mem',
	                     'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                     'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
	                     'AICS(classic)',
	                     'Cellsegm', 'Voronoi', 'Expert1', 'Expert2']
	

	seg_score_dataframe = pd.DataFrame(columns=['seg_score', 'expert', 'method', 'tile'])
	for tile in tile_list:
		for expert in expert_list:
			for method in method_list:
				if expert != method:

					mask_expert = pickle.load(bz2.BZ2File(join(data_dir, tile, 'mask_' + expert + '.pickle'), 'rb'))
					mask_method = pickle.load(bz2.BZ2File(join(data_dir, tile, 'mask_' + method + '.pickle'), 'rb'))
	
					current_seg_score_1 = get_seg_score(mask_expert, mask_method)
					current_seg_score_2 = get_seg_score(mask_method, mask_expert)
					current_seg_score = (current_seg_score_1 + current_seg_score_2) / 2
					if tile == 'R001_X003_Y004':
						tile_name = 'tile1'
					elif tile == 'R001_X004_Y003':
						tile_name = 'tile2'
					method_name = methods_abre_list[method_list.index(method)]
					seg_score_dataframe.loc[len(seg_score_dataframe)] = [current_seg_score, expert, method_name, tile_name]
	
	seg_score_dataframe.to_csv(join(os.path.dirname(data_dir), 'seg_prime_score', 'seg_prime_score.csv'))

