import numpy as np

# package for 3d visualization
from itkwidgets import view
from aicssegmentation.core.visual import seg_fluo_side_by_side,  single_fluorescent_view, segmentation_quick_view
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16, 12]

# package for io
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

# function for core algorithm
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from skimage.morphology import remove_small_objects,  dilation, erosion, ball
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import bz2
import pickle
import os
#%%
from skimage.io import imread, imsave
# import packages
import numpy as np
from os.path import join
import sys


os.getcwd()

file_dir = sys.argv[1]
downsample_percentage = int(sys.argv[2])
pixel_size_in_nano = int(sys.argv[3])

# file_dir = '/data/hubmap/data/MIBI/extracted/Point1'
Data1 = imread(join(file_dir, 'membrane.tif'))
Data2 = imread(join(file_dir, 'cytoplasm.tif'))
# Data = imread(join(file_dir, 'cytoplasm.tif'))
Data = Data1 + Data2

struct_img0 = Data.copy()


################################
## PARAMETERS for this step ##
intensity_scaling_param = [3, 19]
gaussian_smoothing_sigma = 1
################################
# intensity normalization
struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)

# smoothing with 2d gaussian filter slice by slice
structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)


################################
## PARAMETERS for this step ##
s2_param = [[5,0.09], [2.5,0.07], [1,0.01]]
################################

bw_spot = dot_2d_slice_by_slice_wrapper(structure_img_smooth, s2_param)

################################
## PARAMETERS for this step ##
f2_param = [[1, 0.15]]
################################

bw_filament = filament_2d_wrapper(structure_img_smooth, f2_param)

bw = np.logical_or(bw_spot, bw_filament)


################################
## PARAMETERS for this step ##
fill_2d = True
fill_max_size = 1600
# minArea = 15
minRadius = 5 * 1000 / pixel_size_in_nano * downsample_percentage / 100
minArea = np.pi * (minRadius ** 2)
################################

bw_fill = hole_filling(bw, 0, fill_max_size, fill_2d)

seg = remove_small_objects(bw_fill>0, min_size=minArea, connectivity=1, in_place=False)

imsave(join(file_dir, 'mask_aics_lamp.png'), seg)
# np.save(join(file_dir, 'mask_aics_classic.npy'), final_seg)
pickle.dump(seg, bz2.BZ2File(join(file_dir, 'mask_aics_lamp.pickle'), 'wb'))
