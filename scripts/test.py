import numpy as np
import pickle
import bz2
loading_new = np.array([[ 0.11536634 ,0.5644681 ],
 [ 0.09955495,  0.6642034 ],
 [ 0.30956746, -0.39612692],
 [ 0.37383245, -0.10727499],
 [ 0.32295224, -0.15673588],
 [ 0.35239366,  0.11913167],
 [ 0.34437548,  0.10798626],
 [ 0.38448096, -0.07063582],
 [ 0.35736022, -0.03453408],
 [ 0.34437025,  0.12322171]])

loading_old = [[ 0.11265103,  0.61151828],
 [ 0.11478492,  0.68092615],
 [ 0.31129032, -0.34853795],
 [ 0.36297359, -0.07063023],
 [ 0.34691399, -0.11826215],
 [ 0.33645395,  0.09164265],
 [ 0.32831878,  0.08207281],
 [ 0.37924376, -0.02456926],
 [ 0.35122303, -0.07666712],
 [ 0.3700749,   0.01777313]]

def flatten_dict(input_dict):
	local_list = []
	for key, value in input_dict.items():
		if type(value) == dict:
			local_list.extend(flatten_dict(value))
		else:
			local_list.append(value)
	return local_list

metrics = {"Matched Cell": {"NumberOfCellsPer100SquareMicrons": 1.3450115937305542, "FractionOfForegroundOccupiedByCells": 0.8096691677105359, "1-FractionOfBackgroundOccupiedByCells": 0.6992123387179104, "FractionOfCellMaskInForeground": 0.920806368241497, "1/(ln(StandardDeviationOfCellSize)+1)": 0.16357892646300337, "1/(AvgCVForegroundOutsideCells+1)": 0.40852987957780734, "FractionOfFirstPCForegroundOutsideCells": 0.4973791664510944, "1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)": 0.4717248782919162, "AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters": 0.43300338121090787, "AvgSilhouetteOver2~10NumberOfClusters": 0.375962490470272}, "QualityScore": 0.7702643813191148, "PC1": 0.5581920531293383, "PC2": 2.6817319165602065}
pca = pickle.load(open('/Users/hrchen/Downloads/segmentation_RRA/data/output/pca_10_metrics.pickle', 'rb'))
metrics_flat = flatten_dict(metrics)[:-3]
np.matmul(metrics_flat, loading_old)