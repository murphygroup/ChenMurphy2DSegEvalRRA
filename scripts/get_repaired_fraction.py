import glob
import numpy as np

if __name__ == '__main__':
    data_dir_list = glob.glob('/Users/hrchen/Downloads/batch/manuscript_v28_repaired_fraction/**/*txt', recursive=True)
    fraction_list = []
    for data_dir in data_dir_list:
	    fraction = np.loadtxt(data_dir).tolist()
	    fraction_list.append(fraction)
	avg_fraction = np.average(fraction_list)