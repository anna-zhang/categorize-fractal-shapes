# Following an EMD tutorial by Sam Van Kooten: https://gist.github.com/svank/6f3c2d776eea882fd271bba1bd2cc16d

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil

def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    # format image data values to meet requirements of cv2.EMD
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
    return sig

# read in image files to compare
# will need to account for reflected images too
reference_img_fp = 'images/frame.000114.ppm' # reference image filepath
uncategorized_img_fp = 'images/frame.000175.ppm' # uncategorized image filepath

assert os.path.isfile(reference_img_fp), 'file \'{0}\' does not exist'.format(reference_img_fp) # make sure reference image exists at the filepath specified
assert os.path.isfile(uncategorized_img_fp), 'file \'{0}\' does not exist'.format(uncategorized_img_fp) # make sure the uncategorized image exists at the filepath specified
reference_img = cv2.resize(cv2.imread(reference_img_fp, cv2.IMREAD_GRAYSCALE), (150, 150)) # read reference image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
uncategorized_img = cv2.resize(cv2.imread(uncategorized_img_fp, cv2.IMREAD_GRAYSCALE), (150, 150)) # read uncategorized image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel

# print image sizes if they are successfully read
if reference_img is not None:
    print('reference_img.size: ', reference_img.shape)
else:
    print('imread({0}) -> None'.format(reference_img_fp))

if uncategorized_img is not None:
    print('uncategorized_img.size: ', uncategorized_img.shape)
else:
    print('imread({0}) -> None'.format(uncategorized_img_fp))

# write out resized images for reference purposes
cv2.imwrite('output/reference_img_114.png', reference_img)
cv2.imwrite('output/uncategorized_img_175.png', uncategorized_img)

reference_img = reference_img / 255 # change from [0, 255] to [0, 1] values; each white pixel has weight = 1
uncategorized_img = uncategorized_img / 255 # change from [0, 255] to [0, 1] values

# convert image matrices into numpy arrays
arr1 = np.array(reference_img) # convert matrix into a numpy array
arr2 = np.array(uncategorized_img) # convert matrix into a numpy array

# convert numpy arrays of images into signatures for cv2.EMD
sig1 = img_to_sig(arr1) 
sig2 = img_to_sig(arr2)
print(sig1)
print(sig2)

# compute the amount of work required to match smaller weight distribution to larger weight distribution (have all white pixels in the image with fewer white pixels match all of the white pixels in the image with more white pixels)
dist, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)
print(dist) # output the amount of work required to transform one distribution into the other