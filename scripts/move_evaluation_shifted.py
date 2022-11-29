import glob
import os
from os.path import join

if __name__ == '__main__':
	dataset_list = sorted(glob.glob('/media/hrchen/Extreme/segmentation/CODEX/HBM433.MQRQ.278/R001_X006_Y008/random_gaussian_0'))
	dataset_list = dataset_list + sorted(glob.glob('/media/hrchen/Extreme/segmentation/CODEX/HBM988.SNDW.698/R003_X004_Y004/random_gaussian_0'))
	dataset_list = dataset_list + sorted(glob.glob('/media/hrchen/Extreme/segmentation/CODEX/**/R001_X004_Y004/random_gaussian_0'))
	mask_type = 'shifted_mask'
	for dataset in dataset_list:
		ori_dir = join(dataset, mask_type)
		new_dir = '/home/hrchen/Documents/Research/hubmap/script/ChenMurphy2DSegEvalRRA/data/intermediate/manuscript_v28_underseg' + ori_dir[34:]
		new_dir = os.path.dirname(new_dir)
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)
		os.system('cp -r ' + ori_dir + ' ' + new_dir)
		# try:
		# 	os.system('python run_evaluation_standalone.py ' + dataset + ' ' + join(dataset, 'shifted_mask'))
		# except:
		# 	pass