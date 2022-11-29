import numpy as np
from skimage.io import imread
import glob
import os


if __name__ == '__main__':
	img_dir_list = sorted(glob.glob('/media/hrchen/Extreme/segmentation/**/random*/membrane.tif', recursive=True)) + sorted(glob.glob('/media/hrchen/Extreme/segmentation/**/downsampl*/membrane.tif', recursive=True))
	for img_dir in img_dir_list:
		img = imread(img_dir)
		# if not os.path.exists(os.path.join(os.path.dirname(img_dir), 'img_shape.npy')):
		img_dir = os.path.dirname(img_dir)
		img_shape = img.shape
		new_dir1 = '/home/hrchen/Documents/Research/hubmap/script/ChenMurphy2DSegEvalRRA/data/intermediate/manuscript_v28_repaired' + img_dir[34:]
		np.save(os.path.join(new_dir1, 'img_shape.npy'), img_shape)

		new_dir2 = '/home/hrchen/Documents/Research/hubmap/script/ChenMurphy2DSegEvalRRA/data/intermediate/manuscript_v28_nonrepaired' + img_dir[34:]
		np.save(os.path.join(new_dir2, 'img_shape.npy'), img_shape)
