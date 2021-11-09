# Following an EMD tutorial by Sam Van Kooten: https://gist.github.com/svank/6f3c2d776eea882fd271bba1bd2cc16d

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil
import sys # process command line arguments

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


def main():
    # Program usage: python3 emd.py (image folder name) (image 1 number) (image 2 number) (x-resolution) (y-resolution)
    # Read in command line arguments
    num_args = len(sys.argv)
    if num_args != 6:
        # Error
        print("ERROR: Program usage: python3 emd.py (image folder name) (image 1 number) (image 2 number) (x-resolution) (y-resolution)")
        return -1

    folder_name = sys.argv[1] # the name of the folder within the program's directory in which all images to categorize are stored
    reference_image = sys.argv[2] # the file number of the image in the folder to use as the reference image
    uncategorized_image = sys.argv[3] # the file number of the image in the folder to use as the uncategorized image
    x_res = int(sys.argv[4]) # width of image in pixels for resizing
    y_res = int(sys.argv[5]) # height of image in pixels for resizing

    # read in image files to compare
    # will need to account for flips (can use np.fliplr(array)) to get flipped array
    reference_img_fp = folder_name + '/frame.' + '{:0>6}'.format(reference_image) + '.ppm' # reference image filepath
    uncategorized_img_fp = folder_name + '/frame.' + '{:0>6}'.format(uncategorized_image) + '.ppm' # uncategorized image filepath

    assert os.path.isfile(reference_img_fp), 'file \'{0}\' does not exist'.format(reference_img_fp) # make sure reference image exists at the filepath specified
    assert os.path.isfile(uncategorized_img_fp), 'file \'{0}\' does not exist'.format(uncategorized_img_fp) # make sure the uncategorized image exists at the filepath specified
    reference_img = cv2.resize(cv2.imread(reference_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read reference image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
    uncategorized_img = cv2.resize(cv2.imread(uncategorized_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read uncategorized image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel

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
    reference_img_output_fp = 'output/frame.' + '{:0>6}'.format(reference_image) + '.png' # output reference image filepath
    uncategorized_img_output_fp = 'output/frame.' + '{:0>6}'.format(uncategorized_image) + '.png' # output uncategorized image filepath
    cv2.imwrite(reference_img_output_fp, reference_img)
    cv2.imwrite(uncategorized_img_output_fp, uncategorized_img)

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
        

if __name__ == "__main__":
    main()