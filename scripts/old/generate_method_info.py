import pandas as pd

if __name__ == '__main__':
	methods = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	           'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0', 'cellpose_new',
	           'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial']
	
	methods_abre_nonrepaired = ['DeepCell 0.12.3 cell membrane', 'DeepCell 0.12.3 cytoplasm',
	                            'DeepCell 0.9.0 cell membrane',
	                            'DeepCell 0.9.0 cytoplasm', 'DeepCell 0.6.0 cell membrane', 'DeepCell 0.6.0 cytoplasm',
	                            'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX',
	                            'AICS(classic)',
	                            'Cellsegm', 'Voronoi']
	
	methods_abre_repaired = ['DeepCell 0.12.3 cell membrane+r', 'DeepCell 0.12.3 cytoplasm+r',
	                         'DeepCell 0.9.0 cell membrane+r',
	                         'DeepCell 0.9.0 cytoplasm+r', 'DeepCell 0.6.0 cell membrane+r',
	                         'DeepCell 0.6.0 cytoplasm+r',
	                         'Cellpose 2.1.0+r', 'Cellpose 0.6.1+r', 'Cellpose 0.0.3.1+r', 'CellProfiler+r', 'CellX+r',
	                         'AICS(classic)+r',
	                         'Cellsegm+r', 'Voronoi+r']
	
	method_names = pd.DataFrame('')
	
	cmap_repaired = ['darkorchid', 'darkturquoise', 'deepskyblue', 'darkviolet', 'darkred', 'deeppink', 'seagreen',
	                 'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan']
	cmap_nonrepaired = ['orchid', 'turquoise', 'skyblue', 'violet', 'red', 'lightpink', 'darkseagreen', 'limegreen',
	                    'khaki', 'slateblue', 'lightsalmon', 'sandybrown', 'goldenrod', 'cyan']
	cmap = cmap_nonrepaired + cmap_repaired
	
	marker_nonrepaired = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o"]
	marker_repaired = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o"]
	marker = marker_nonrepaired + marker_repaired

