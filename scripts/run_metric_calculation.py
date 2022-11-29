import os
from os.path import join
import glob

if __name__ == '__main__':
	methods = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new', 'deepcell_cytoplasm_new', 'deepcell_membrane',  'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new', 'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
	file_dir = os.getcwd()
	data_dir = join(file_dir, 'data', 'intermediate', 'manuscript_v28_mask')
	script_dir = join(file_dir, 'scripts')
	mask_dir_list = sorted(glob.glob(join(data_dir, '**', 'random*'), recursive=True)) + sorted(glob.glob(join(data_dir, '**', 'down*'), recursive=True))
	for mask_dir in mask_dir_list[:1]:
		for method in methods[:1]:
			os.system('python ' + script_dir + '/mask_processing.py ' + mask_dir + ' ' + method + ' ' + 'repaired')
			os.system('python ' + script_dir + '/metric_calculation.py ' + mask_dir + ' ' + method + ' ' + 'repaired')
			os.system('python ' + script_dir + '/mask_processing.py ' + mask_dir + ' ' + method + ' ' + 'nonrepaired')
			os.system('python ' + script_dir + '/metric_calculation.py ' + mask_dir + ' ' + method + ' ' + 'nonrepaired')
			
	
