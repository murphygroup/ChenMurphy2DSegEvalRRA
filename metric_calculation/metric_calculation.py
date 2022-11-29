import numpy as np
import os
from os.path import join
import sys
# from PIL import ImageChops
from scipy.sparse import csr_matrix
from skimage.io import imread
import bz2
import pickle
import cv2
from sklearn.preprocessing import normalize, scale, StandardScaler, MinMaxScaler
# import matplotlib.pyplot as plt
# from sklearn.metrics import jaccard_score as JI
# from skimage import metrics
# from scipy.stats import kurtosis
# from scipy.integrate import simps
# from scipy.stats import ks_2samp
# from scipy.signal import find_peaks
# from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import variation
# from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# import pandas as pd
# from skimage.segmentation import chan_vese
from skimage.morphology import disk
# from skimage.morphology import square
from skimage.morphology import closing, area_closing
from skimage.morphology import diameter_closing
from skimage.filters import threshold_otsu as otsu
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
import sklearn
print(sklearn.__version__)
# import cv2
import scipy
# from skimage.filters import gaussian

def fraction(img_bi, mask_bi):
	foreground_all = np.sum(img_bi)
	background_all = img_bi.shape[0] * img_bi.shape[1] - foreground_all
	mask_all = np.sum(mask_bi)
	background = len(np.where(mask_bi - img_bi == 1)[0])
	foreground = np.sum(mask_bi * img_bi)
	if background_all == 0:
		background_fraction = 0
	else:
		background_fraction = background / background_all	
	# for i in range(0, img_bi.shape[0]):
	# 	for j in range(0, img_bi.shape[1]):
	# 		if img_bi[i, j] == 0 and mask_bi[i, j] == 1:
	# 			background += 1
	# 		elif img_bi[i, j] == 1 and mask_bi[i, j] == 1:
	# 			foreground += 1
	
	return foreground / foreground_all, background_fraction, foreground / mask_all

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def thresholding(img):
	from skimage.filters import threshold_mean
	threshold = threshold_mean(img.astype(np.int64))
	img_thre = img > threshold
	img_thre = img_thre * 1
	return img_thre


def foreground_separation(img_thre):
	from skimage.filters import threshold_mean
	from skimage import measure
	#np.savetxt(join(tile_dir, 'image_original.txt'), img)
	#threshold = threshold_mean(img.astype(np.int64))
	#img_sep = img > threshold
	plt.imsave(join(tile_dir, 'image_thresholded.png'), img_thre)
	contour_ref = img_thre.copy()	
	img_thre = closing(img_thre, disk(1))

	img_thre = -img_thre + 1
	img_thre = closing(img_thre, disk(1))
	img_thre = -img_thre + 1

	img_thre = closing(img_thre, disk(10))

	img_thre = -img_thre + 1
	img_thre = closing(img_thre, disk(10))
	img_thre = -img_thre + 1

	img_thre = area_closing(img_thre, 5000, connectivity=2)
	contour_ref = contour_ref.astype(float)
	img_thre = img_thre.astype(float)
	img_binary = MorphGAC(-contour_ref+1, 5, -img_thre+1, smoothing=1, balloon=0.8, threshold=0.5)
	img_binary = area_closing(img_binary, 1000, connectivity=2)
	

	return -img_binary+1



def uniformity_CV(loc):
	CV = []
	n = len(channels)
	for i in range(n):
		channel = imread(join(channel_dir, channels[i]))
		channel = channel / np.mean(channel)
		intensity = channel[tuple(loc.T)]
		CV.append(np.std(intensity))
	return np.average(CV)

def uniformity_fraction(loc):
	n = len(channels)
	ss = StandardScaler()
	for i in range(n):
		# channel = imread(os.path.join(data_dir, 'R001_X001_Y001.ome-' + str(i) + '.tif'))
		channel = imread(join(channel_dir, channels[i]))
		channel = ss.fit_transform(channel.copy())
		intensity = channel[tuple(loc.T)]
		if i == 0:
			feature_matrix = intensity
		else:
			feature_matrix = np.vstack((feature_matrix, intensity))
	# print(feature_matrix.shape)
	pca = PCA(n_components=1)
	model = pca.fit(feature_matrix.T)
	fraction = model.explained_variance_ratio_[0]
	return fraction

def foreground_uniformity(img_bi, mask):
	foreground_loc = np.argwhere((img_bi - mask) == 1)
	# foreground_loc = np.argwhere(mask == 0)
	# foreground_loc = np.array([x for x in set(tuple(x) for x in np.argwhere(img_bi == 1)) & set(tuple(x) for x in np.argwhere(mask == 0))])
	CV = uniformity_CV(foreground_loc)
	fraction = uniformity_fraction(foreground_loc)
	return CV, fraction

def background_uniformity(img_bi, mask):
	background_loc = np.argwhere(img_bi == 0)
	# background_loc = np.array([x for x in set(tuple(x) for x in np.argwhere(img_bi == 0)) & set(tuple(x) for x in np.argwhere(mask == 0))])
	# print(len(background_loc_test))
	# print(len(background_loc))
	# print((background_loc_test  == background_loc).all())
	# background_loc = np.array([x for x in set(tuple(x) for x in np.argwhere(img_bi == 1)) & set(tuple(x) for x in np.argwhere(mask == 0))])
	if len(background_loc) == 0:
		CV = 0
		fraction = 1
	else:
		CV = uniformity_CV(background_loc)
		fraction = uniformity_fraction(background_loc)
	return CV, fraction


def cell_uniformity_CV(feature_matrix):
	CV = []
	for i in range(feature_matrix.shape[1]):
		# print(feature_matrix[:, i].shape)
		if np.sum(feature_matrix[:, i]) == 0:
			CV.append(np.nan)
		else:
			CV.append(variation(feature_matrix[:, i]))
		# if feature_matrix.shape[0] == 2:
		# 	print(feature_matrix.shape)
	if np.sum(np.nan_to_num(CV)) == 0:
		return 0
	else:
		return np.nanmean(CV)



def cell_uniformity_fraction(feature_matrix):
	# print(feature_matrix.shape)
	if np.sum(feature_matrix) == 0 or feature_matrix.shape[0] == 1:
		return 1
	else:
		pca = PCA(n_components=1)
		model = pca.fit(feature_matrix)
		# print(pca.fit_transform(ss.fit_transform(feature_matrix)).shape)
		fraction = model.explained_variance_ratio_[0]
		return fraction

def weighted_by_cluster(vector, labels):
	for i in range(len(vector)):
		# print(vector[i])
		# print(len(np.where(labels == i)[0]))
		vector[i] = vector[i] * len(np.where(labels == i)[0])
	weighted_average = np.sum(vector) / len(labels)
	return weighted_average

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def cell_uniformity(mask, read_kmeans):
	# cell_index = int(np.loadtxt(join(os.path.dirname(result_dir), 'result_' + repair_mask + '_cell_matched', 'cell_basic_' + method + '.txt'))[0])
	cell_num = len(np.unique(mask)) - 1
	n = len(channels)
	cell_coord = get_indices_sparse(mask)[1:]
	cell_coord_num = len(cell_coord)
	ss = StandardScaler()
	for i in range(n):
		channel = imread(join(channel_dir, channels[i]))
		channel_z = ss.fit_transform(channel)
		# channel = channel / np.mean(channel)
		cell_intensity = []
		cell_intensity_z = []
		empty_cell_list = []
		cell_size = []
		for j in range(cell_coord_num):
			cell_size_current = len(cell_coord[j][0])
			if cell_size_current == 0:
				# single_cell_intensity = 0
				empty_cell_list.append(j)
			else:
				single_cell_intensity = np.sum(channel[tuple(cell_coord[j])]) / cell_size_current
				single_cell_intensity_z = np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
				cell_intensity.append(single_cell_intensity)
				cell_intensity_z.append(single_cell_intensity_z)
				if i == 0:
					cell_size.append(cell_size_current)
		if i == 0:
			feature_matrix = np.array(cell_intensity)
			feature_matrix_z = np.array(cell_intensity_z)
			cell_size_list = np.expand_dims(np.array(cell_size), 1)
		else:
			feature_matrix = np.vstack((feature_matrix, cell_intensity))
			feature_matrix_z = np.vstack((feature_matrix_z, cell_intensity_z))
	# print(empty_cell_list)
	feature_matrix = feature_matrix.T
	feature_matrix_z = feature_matrix_z.T
	CV = []
	fraction = []
	silhouette = []
	silhouette_cell_size = []
	#for c in range(1, (np.ceil(cell_num / 100).astype(int)+1)):
	for c in range(1, 11):
		try:
			# print(c)
			if not read_kmeans:
				model = KMeans(n_clusters=c, random_state=3).fit(feature_matrix_z)
				labels = model.labels_.astype(int)
				np.savetxt(join(result_dir, 'kmeans_labels_' + str(c) + '_' + method + '.txt'), [labels])
			else:
				labels = np.loadtxt(join(os.path.dirname(result_dir), 'result_' + repair_mask + '_cell_matched', 'kmeans_labels_' + str(c) + '_' + method + '.txt'))
				# print(len(labels_original))
				# if len(empty_cell_list) == 0:
				# 	labels = labels_original
				# else:
				# 	labels = np.array([int(i) for j, i in enumerate(labels_original) if j not in empty_cell_list])
			# print(len(labels))
			CV_current = []
			fraction_current = []

			
			for i in range(c):
				cluster_feature_matrix = feature_matrix[np.where(labels == i)[0], :]
				cluster_feature_matrix_z = feature_matrix_z[np.where(labels == i)[0], :]
				# print(cluster_feature_matrix.shape)
				# print(cluster_feature_matrix)
				# print(cluster_feature_matrix.shape)
				CV_current.append(cell_uniformity_CV(cluster_feature_matrix))
				fraction_current.append(cell_uniformity_fraction(cluster_feature_matrix_z))

			# except:
			# 	silhouette.append(0)
			# print(CV_current)
			CV.append(weighted_by_cluster(CV_current, labels))
			fraction.append(weighted_by_cluster(fraction_current, labels))
			
			if c == 1:
				silhouette.append(0)
				silhouette_cell_size.append(0)
			else:
				# try:
				# print(feature_matrix_z.shape)
				silhouette.append(silhouette_score(feature_matrix_z, labels))
				silhouette_cell_size.append(silhouette_score(cell_size_list, labels))
		except:
			if len(CV) != c:
				CV.append(0)
			if len(fraction) != c:
				fraction.append(0)
			if len(silhouette) != c:
				silhouette.append(-1)
			if len(silhouette_cell_size) != c:
				silhouette_cell_size.append(-1)
	cell_size_std = np.std(cell_size_list)
	return CV, fraction, silhouette, silhouette_cell_size, cell_size_std



if __name__ == '__main__':
	print('Calculating evaluation metrics..')
	file_dir = sys.argv[1]
	repair_mask = sys.argv[3]
	if repair_mask == 'repaired':
		matched_mask_dir = join(file_dir, 'repaired_mask')
	elif repair_mask == 'nonrepaired':
		matched_mask_dir = file_dir
	method = sys.argv[2]
	compartment_list = ['cell_matched', 'nuclear_matched', 'cell_outside_nucleus_matched']
	# compartment_list = ['cell_matched']
	compartment_list_output = ['matched cell mask', 'matched nuclear mask', 'matched cell outside nucleus mask']
	# compartment_list_output = ['matched cell mask']
	for compartment_ind in range(len(compartment_list)):
		compartment = compartment_list[compartment_ind]
		print(compartment_list_output[compartment_ind] + '...')
		result_dir = join(file_dir, 'result_' + repair_mask + '_' + compartment)
		if not os.path.exists(result_dir):
			os.makedirs(result_dir)
			
		# determine the noise type and dirs
		tile_dir = os.path.split(file_dir)[0]
		tile_name = os.path.split(os.path.split(file_dir)[0])[1]
		file_name = os.path.split(file_dir)[1]
		if file_name[:15] == 'random_gaussian':
			noise_type = 'gaussian'
			channel_dir = join(tile_dir, 'channels')
			img_binary_dir = join(tile_dir, 'image_binary.txt')
			img_dir = tile_dir
		else:
			noise_type = 'downsampling'
			channel_dir = join(tile_dir, 'channels_' + file_name)
			img_binary_dir = join(tile_dir, 'image_binary_' + file_name + '.txt')
			img_dir = file_dir
			
		if compartment == 'cell_matched':
			
			# calculate fraction between cell and nuclear mask
			mask_dir = bz2.BZ2File(join(file_dir, 'mask_' + method + '.pickle'), 'rb')
			mask = pickle.load(mask_dir)
			mask = mask.astype(int)
			try:
				nuclear_mask_dir = bz2.BZ2File(join(file_dir, 'nuclear_mask_' + method + '.pickle'), 'rb')
				nuclear_mask = pickle.load(nuclear_mask_dir)
			except:
				nuclear_mask_dir = bz2.BZ2File(join(file_dir, 'nuclear_mask_artificial.pickle'), 'rb')
				nuclear_mask = pickle.load(nuclear_mask_dir)
			cell_matched_mask_dir = bz2.BZ2File(join(matched_mask_dir, 'cell_matched_mask_' + method + '.pickle'), 'rb')
			cell_matched_mask = pickle.load(cell_matched_mask_dir)
	
			nuclear_matched_mask_dir = bz2.BZ2File(join(matched_mask_dir, 'nuclear_matched_mask_' + method + '.pickle'), 'rb')
			nuclear_matched_mask = pickle.load(nuclear_matched_mask_dir)
			
			if repair_mask == 'repaired':
				fraction_matched_cells = 1
			elif repair_mask == 'nonrepaired':
				matched_cell_num = len(np.unique(cell_matched_mask))
				total_cell_num = len(np.unique(mask))
				total_nuclei_num = len(np.unique(nuclear_mask))
				mismatched_cell_num = total_cell_num - matched_cell_num
				mismatched_nuclei_num = total_nuclei_num - matched_cell_num
				fraction_matched_cells = matched_cell_num / (mismatched_cell_num + mismatched_nuclei_num + matched_cell_num)
			np.savetxt(join(result_dir, 'metric_fraction_matched_cell_nuclei_num_' + method + '.txt'), [fraction_matched_cells])
			mask = cell_matched_mask.astype(int)
			
			# foreground-background separation
			
			if noise_type == 'gaussian':
				if not os.path.exists(img_binary_dir) and file_name == 'random_gaussian_0':
				#if file_name == 'random_gaussian_0':
					print('Separating image foreground and background...')
					img = np.dstack((imread(join(img_dir, 'random_gaussian_0', 'nucleus.tif')), imread(join(img_dir, 'random_gaussian_0', 'cytoplasm.tif')), imread(join(img_dir, 'random_gaussian_0', 'membrane.tif'))))
					img_thresholded = sum(thresholding(img[:, :, c]) * 1 for c in range(img.shape[2]))
					img_thresholded = np.sign(img_thresholded).astype(int)
					img_binary = foreground_separation(img_thresholded)
					img_binary = np.sign(img_binary)
					plt.imsave(join(tile_dir, img_binary_dir[:-4] + '.png'), img_binary)
					np.savetxt(img_binary_dir, img_binary)
					del img
				else:
					img_binary = np.loadtxt(img_binary_dir)
			elif noise_type == 'downsampling':
				img_binary = np.loadtxt(join(tile_dir, 'image_binary.txt'))
				x = mask.shape[0]
				y = mask.shape[1]
				img_binary_downsampled = cv2.resize(img_binary, (y, x), interpolation=cv2.INTER_AREA)
				img_binary_downsampled[np.where(img_binary_downsampled < 0)] = 0
				img_binary_downsampled[np.where(img_binary_downsampled > 65535)] = 65535
				img_binary_downsampled = np.sign(img_binary_downsampled)
				# img_binary_downsampled = img_binary_downsampled.astype('uint16')
				plt.imsave(join(tile_dir, img_binary_dir[:-4] + '.png'), img_binary_downsampled)
				np.savetxt(join(tile_dir, img_binary_dir), img_binary_downsampled)
				img_binary = img_binary_downsampled
			read_kmeans_labels = False
			
		else:
			if file_name[:15] == 'random_gaussian':
				channel_dir = join(tile_dir, 'channels')
				img_binary = np.loadtxt(join(tile_dir, 'image_binary.txt'))
			else:
				channel_dir = join(tile_dir, 'channels_' + file_name)
				img_binary = np.loadtxt(join(tile_dir, 'image_binary_' + file_name + '.txt'))
			mask_dir = bz2.BZ2File(join(matched_mask_dir, compartment + '_mask_' + sys.argv[2] + '.pickle'), 'rb')
			mask = pickle.load(mask_dir)
			mask = mask.astype(int)
			read_kmeans_labels = True
	
		# error if empty mask
		mask_binary = np.sign(mask)
		if np.sum(mask_binary) == 0:
			raise ValueError('invalid mask')
		
		# coverage metrics
		cell_num = len(np.unique(mask)) - 1
		mask_fraction = 1 - (len(np.where(mask == 0)[0]) / (mask.shape[0] * mask.shape[1]))
		foreground_fraction, background_fraction, mask_foreground_fraction = fraction(img_binary, mask_binary)
		
		# homogeneity metrics
		channels = os.listdir(channel_dir)
		foreground_CV, foreground_PCA = foreground_uniformity(img_binary, mask)
		np.savetxt(join(result_dir, 'metric_foreground_' + sys.argv[2] + '.txt'), [foreground_CV, foreground_PCA])

		background_CV, background_PCA = background_uniformity(img_binary, mask)
		np.savetxt(join(result_dir, 'metric_background_' + sys.argv[2] + '.txt'), [background_CV, background_PCA])
		
		cell_CV, cell_fraction, cell_silhouette, cell_size_silhouette, cell_size_std = cell_uniformity(mask, read_kmeans_labels)
		#print(cell_CV)
		#print(cell_fraction)
		#print(cell_silhouette)
		#print(cell_size_silhouette)
		#print(cell_size_std)
		#print([cell_num, mask_fraction, foreground_fraction, background_fraction, mask_foreground_fraction, cell_size_std])
		np.savetxt(join(result_dir, 'metric_cell_basic_' + sys.argv[2] + '.txt'), [cell_num, mask_fraction, foreground_fraction, background_fraction, mask_foreground_fraction, cell_size_std])
		np.savetxt(join(result_dir, 'metric_cell_' + sys.argv[2] + '.txt'), [cell_CV, cell_fraction, cell_silhouette, cell_size_silhouette])


