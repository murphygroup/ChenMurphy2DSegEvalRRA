import json
import os

import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import gridspec
import matplotlib.path as mpath

if __name__ == '__main__':
	file_dir = os.getcwd()
	
	method_list = ['deepcell_membrane-0.12.3', 'deepcell_cytoplasm-0.12.3', 'deepcell_membrane_new',
	               'deepcell_cytoplasm_new', 'deepcell_membrane', 'deepcell_cytoplasm', 'cellpose-2.1.0',
	               'cellpose_new',
	               'cellpose', 'cellprofiler', 'CellX', 'aics_classic', 'cellsegm', 'artificial', 'expert1', 'expert2']
	methods_abre = ['DeepCell 0.12.3 mem', 'DeepCell 0.12.3 cyto', 'DeepCell 0.9.0 mem',
	                'DeepCell 0.9.0 cyto', 'DeepCell 0.6.0 mem', 'DeepCell 0.6.0 cyto',
	                'Cellpose 2.1.0', 'Cellpose 0.6.1', 'Cellpose 0.0.3.1', 'CellProfiler', 'CellX', 'AICS(classic)',
	                'Cellsegm', 'Voronoi', 'Expert1', 'Expert2']
	
	mask_dir = join(file_dir, 'data', 'annotation', 'CODEX', 'HBM279.TQRS.775')
	print(mask_dir)
	quality_score_list = []
	pc1_list = []
	pc2_list = []
	for method in method_list:
		with open(join(mask_dir, 'R001_X003_Y004', 'metrics_' + method + '_v28.json'), 'r') as f:
			metrics = json.load(f)
		quality_score = metrics['QualityScore']
		pc1 = metrics['PC1']
		pc2 = metrics['PC2']
		with open(join(mask_dir, 'R001_X004_Y003', 'metrics_' + method + '_v28.json'), 'r') as f:
			metrics = json.load(f)
		quality_score = (quality_score + metrics['QualityScore']) / 2
		pc1 = (pc1 + metrics['PC1']) / 2
		pc2 = (pc2 + metrics['PC2']) / 2
		quality_score_list.append(quality_score)
		pc1_list.append(pc1)
		pc2_list.append(pc2)
	
	df = pd.DataFrame(
		dict(
			quality_score=quality_score_list,
			methods=methods_abre
		)
	)
	
	df_sorted = df.sort_values('quality_score', ascending=False)
	
	y_pos = np.arange(len(methods_abre))
	
	fig = plt.figure()
	fig.set_figheight(5)
	fig.set_figwidth(15)
	
	gs = gridspec.GridSpec(6, 6, left=0.45, right=0.98, hspace=0.25, wspace=1)
	ax1 = fig.add_subplot(gs[:, :3])
	ax2 = fig.add_subplot(gs[:-2, 3:])
	
	barlist = ax1.barh('methods', 'quality_score', data=df_sorted)
	
	ax1.invert_yaxis()  # labels read top-to-bottom
	ax1.set_xlabel('Quality Score', fontsize=13)
	ax1.tick_params(axis='both', which='major', labelsize=13)
	barlist[4].set_color('g')
	barlist[11].set_color('g')
	
	cmap_repaired = ['darkorchid', 'darkturquoise', 'deepskyblue', 'darkviolet', 'darkred', 'deeppink', 'seagreen',
	                 'darkgreen', 'darkkhaki', 'darkslateblue', 'darksalmon', 'chocolate', 'darkgoldenrod', 'darkcyan',
	                 'dimgrey', 'grey']
	cmap = cmap_repaired
	star = mpath.Path.unit_regular_star(6)
	
	# marker_repaired = ["o", "s", "^", "P", "D", "*", "X", "h", "v", "d", "p", "$E$", "$E$"]
	marker_repaired = [star, "*", "X", "P", "^", "v", "H", "h", "p", "8", 'D', "s", "d", "o", '<', '>']
	marker = marker_repaired
	
	for i in range(len(methods_abre)):
		ax2.scatter(pc1_list[i], pc2_list[i], color=cmap[i],
		            s=60, marker=marker[i], label=methods_abre[i])
	
	ax2.set_xlabel('PC1 (57%)', fontsize=12)
	ax2.set_ylabel('PC2 (17%)', fontsize=12)
	ax2.tick_params(axis='both', which='major', labelsize=13)

	ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fancybox=True, shadow=True, ncol=2)
	
	# fig.tight_layout()
	print(join(file_dir, 'figures', 'annotation_top2PC_rankings.png'))
	plt.savefig(join(file_dir, 'figures', 'annotation_top2PC_rankings.png'), bbox_inches='tight', dpi=500)
	plt.clf()
	plt.close()
