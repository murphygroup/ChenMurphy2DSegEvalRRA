import glob
import os
from os.path import join

if __name__ == '__main__':
	dataset_list = sorted(glob.glob('/Volumes/Extreme/segmentation/CODEX/HBM433.MQRQ.278/R001_X006_Y008/random_gaussian_0'))
	dataset_list = dataset_list + sorted(glob.glob('/Volumes/Extreme/segmentation/CODEX/HBM988.SNDW.698/R003_X004_Y004/random_gaussian_0'))
	dataset_list = dataset_list + sorted(glob.glob('/Volumes/Extreme/segmentation/CODEX/**/R001_X004_Y004/random_gaussian_0'))
	for dataset in dataset_list:
		# try:
			os.system('python run_evaluation_standalone.py ' + dataset + ' ' + join(dataset, 'merged_fraction_mask'))
		# except:
		# 	pass