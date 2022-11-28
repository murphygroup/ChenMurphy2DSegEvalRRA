import os

import numpy as np





if __name__ == '__main__':
	script_dir = '/home/haoranch/projects/HuBMAP'

	os.system('srun -p pool1 python ' + script_dir + '/seg_prime_score.py &')

	jaccard_list = np.linspace(0, 1, 101)
	for jaccard in jaccard_list:
		os.system('srun -p pool1 python ' + script_dir + '/f1_score.py ' + str(jaccard) + ' &')
	

		


