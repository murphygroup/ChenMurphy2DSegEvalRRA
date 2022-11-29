import numpy as np
from os.path import join
import os
import pandas as pd
import glob
import cv2 as cv
import json
from skimage.io import imread
import gdown
import sys
import requests

def make_dir(save_dir):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
def laplacian_variance(img):
	return np.var(cv.Laplacian(img, cv.CV_64F, ksize=21))


def extend_str_zero(string, target):
	while len(string) != target:
		string = '0' + string
	return string

def get_raw_tif(TMC, tile_idx, part, idx, img_id, tissue_id, save_dir, tile_name, max_var_id=None):
	file_structure_img = file_structure[img_id][tissue_id]
	folder_names = list(file_structure_img.keys())
	cyc = str(idx // 4 + 1)
	if TMC == 'Florida':
		cyc = extend_str_zero(cyc, 3)
		folder_name_partial = 'cyc' + str(cyc) + '_reg001'
		folder_name = [i for i in folder_names if i.startswith(folder_name_partial)][0]
	else:
		folder_name = 'Cyc' + str(cyc) + '_' + 'reg' + str(R)
	
	ch = str(idx % 4)
	if ch == '0':
		ch = '4'
	tile_idx = extend_str_zero(tile_idx, 5)
	result_dir = join(save_dir, tile_name)
	make_dir(result_dir)
	if part == 'nucleus':
		for z in range(30):
			tif_file_name = '1_' + tile_idx + '_Z' + extend_str_zero(str(z), 3) + '_CH' + ch + '.tif'
			try:
				download_path = join('https://g-d00e7b.09193a.5898.dn.glob.us', img_id, tissue_id, folder_name, tif_file_name)
				os.system('wget -P ' +  result_dir + ' ' + download_path)
			except:
				pass
			
		slice_list = sorted(glob.glob(join(result_dir, '*CH*.tif')))
		if len(slice_list) > 0:
			lap_vars_per_z_plane = []
			for slice in slice_list:
				try:
					lap_vars_per_z_plane.append(laplacian_variance(imread(slice)))
				except:
					lap_vars_per_z_plane.append(0)
			max_var = max(lap_vars_per_z_plane)
			max_var_id = str(lap_vars_per_z_plane.index(max_var)+1)
			max_var_id = extend_str_zero(max_var_id, 3)
			os.system('mv ' + join(result_dir, '*Z' + max_var_id + '*CH*.tif') + ' ' + join(result_dir, part + '.tif'))
			os.system('rm ' + join(result_dir, '*CH*tif'))
			return max_var_id
		else:
			os.system('rm -rf ' + result_dir)
			return False
	else:
		tif_file_name = '1_' + tile_idx + '_Z' + extend_str_zero(str(max_var_id), 3) + '_CH' + ch + '.tif'
		download_path = join('https://g-d00e7b.09193a.5898.dn.glob.us', img_id, tissue_id, folder_name, tif_file_name)
		os.system('wget -P ' + result_dir + ' ' + download_path)
		
		os.system('chmod 777 *tif')
		os.system('mv ' + join(result_dir, '*Z' + max_var_id + '*CH*.tif') + ' ' + join(result_dir, part + '.tif'))


def get_CODEX_data(output_dir):
	current_dir = os.path.dirname(os.path.realpath(__file__))
	checklist = np.loadtxt(join(current_dir, 'CODEX_img_list.txt'), dtype=str)
	global file_structure
	file_structure = json.load(open(join(current_dir, 'CODEX_file_structure.json'), 'r'))
	for dataset in checklist:
		img_name = dataset[0]
		img_id = dataset[1]
		tissue_id = dataset[2]
		R_num = int(dataset[4])
		X_num = int(dataset[5])
		Y_num = int(dataset[6])
		if img_name == 'HBM433.MQRQ.278' or img_name == 'HBM988.SNDW.698':
			TMC = 'Standford'
		else:
			TMC = 'Florida'
		save_dir = join(output_dir, img_name)
		make_dir(save_dir)
		os.system('wget -P ' + save_dir + ' ' + join('https://g-d00e7b.09193a.5898.dn.glob.us', img_id, tissue_id, 'channelnames_report.csv'))
		channel_info = pd.read_csv(glob.glob(join(save_dir, '*.csv'))[0], header=None).iloc[:, 0].values.tolist()
		if TMC == 'Florida':
			nucleus_idx = channel_info.index('DAPI-02') + 1
			cytoplasm_idx = channel_info.index('CD107a') + 1
			membrane_idx = channel_info.index('E-CAD') + 1
		else:
			nucleus_idx = channel_info.index('HOECHST1') + 1
			cytoplasm_idx = channel_info.index('cytokeratin') + 1
			membrane_idx = channel_info.index('CD45') + 1
		shared_channels = [channel_info.index('CD11c') + 1, channel_info.index('CD21') + 1, channel_info.index('CD4') + 1, channel_info.index('CD8') + 1, channel_info.index('Ki67') + 1]
		global R
		R = 1
		while R <= R_num:
			Y = 2
			while Y <= Y_num and Y < 10:
				X = 2
				while X <= X_num and X < 10:
					if Y % 2 == 1:
						r_idx = str(X + (Y-1) * X_num)
					else:
						r_idx = str(X_num - (X-1) + (Y-1) * X_num)
					tile_name = 'R00' + str(R) + '_X00' + str(X) + '_Y00' + str(Y)
					if not os.path.exists(join(save_dir, tile_name, 'nucleus.tif')):
						z_slice_id = get_raw_tif(TMC, r_idx, 'nucleus', nucleus_idx, img_id, tissue_id, save_dir, tile_name)
						if z_slice_id:
							get_raw_tif(TMC, r_idx, 'cytoplasm', cytoplasm_idx, img_id, tissue_id, save_dir, tile_name, z_slice_id)
							get_raw_tif(TMC, r_idx, 'membrane', membrane_idx, img_id, tissue_id, save_dir, tile_name, z_slice_id)
							for channel_idx in shared_channels:
								# get_raw_tif(r_idx, 'CH_' + channel_info[channel_idx-1], channel_idx, id, save_dir, tile_name, z_slice_id)
								get_raw_tif(TMC, r_idx, 'CH_' + str(channel_idx), channel_idx, img_id, tissue_id, save_dir, tile_name, z_slice_id)
							channel_dir = join(save_dir, tile_name, 'channels')
							if not os.path.exists(channel_dir):
								os.makedirs(channel_dir)
							os.system('mv ' + join(save_dir, tile_name, 'CH*tif') + ' ' + channel_dir)
					X = X + 2
				Y = Y + 2
			R = R + 1

def get_IMC_data(output_dir):
	current_dir = os.path.dirname(os.path.realpath(__file__))
	checklist = np.loadtxt(join(current_dir, 'IMC_img_list.txt'), dtype=str)
	for dataset in checklist:
		img_name = dataset[0]
		img_id = dataset[1]
		tissue_id = dataset[3]
		tif_file_name = dataset[4]
		save_dir = join(output_dir, img_name)
		make_dir(save_dir)
		os.system('wget -P ' + save_dir + ' ' + join('https://g-d00e7b.09193a.5898.dn.glob.us', img_id, 'ometiff', tissue_id, tif_file_name))


if __name__ == '__main__':
	file_dir = os.getcwd()
	save_dir = join(file_dir, 'data')
	get_CODEX_data(join(save_dir, 'CODEX'))
	get_IMC_data(join(save_dir, 'IMC'))

