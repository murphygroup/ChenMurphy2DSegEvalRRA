import numpy as np
from skimage.io import imread
import glob
import os


if __name__ == '__main__':
	img_dir_list = sorted(glob.glob('/Volumes/Extreme/segmentation/**/nucleus.tif', recursive=True))
	for img_dir in img_dir_list:
		print(img_dir)
		img_dir = '/Volumes/Extreme/segmentation/CellDIVE/region_all/S20030077_region_006/X3_Y3/downsampling_30/membrane.tif'
		try:
			img = imread(img_dir)
			if not os.path.exists(os.path.join(os.path.dirname(img_dir), 'img_shape.npy')):
				img_shape = img.shape
				np.save(os.path.join(os.path.dirname(img_dir), 'img_shape.npy'), img_shape)
		except:
			pass
	
	
			