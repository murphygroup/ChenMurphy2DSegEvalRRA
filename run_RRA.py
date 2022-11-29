import json
import os
from os.path import join
# from pipeline.data.download_HuBMAP_data import *
# from pipeline.segmentation.methods.installation.install_all_methods import *

if __name__ == '__main__':
	os.system('python image_download/download_HuBMAP_data.py')
	os.system('python segmentation/run_segmentation.py')
	os.system('python metric_calculation/run_metric_calculation.py')
	os.system('bash result_analysis/reproduce_all_fig.sh')


