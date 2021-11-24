import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil
import sys # process command line arguments
import math
import datetime

class MatchInfo:
  def __init__(self, match, score, error):
    self.match = match # True if the two shapes match, False if the two shapes do not match
    self.score = score # match score between the two shapes
    self.error = error # True if there was a computation error, False otherwise


# computes ratio of white pixels to total pixels, if less than a threshold value, then considers the image shapeless
# returns False if the ratio of white pixels:total pixels is lower than the threshold value, True otherwise
def shape_exists(img, white_threshold): 
    img = img / 255.0

    num_white_pixels = 0 # number of white pixels in the image
    height, width = img.shape
    total_pixels = height * width # total number of pixels in the image

    for y in range(height):
        for x in range(width):
            if img[y][x] == 1.0:
                num_white_pixels += 1

    white_ratio = float(num_white_pixels) / float(total_pixels) # get ratio of white pixels to all pixels in the image
    if (white_ratio < white_threshold):
        return False
    else:
        return True


def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    # format image data values to meet requirements of cv2.EMD
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    nonzero_value = False
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
            if arr[i,j] > 0:
                nonzero_value = True # a non-zero value in the signature
    if nonzero_value:
        return sig
    else:
        return None

# compute Earth mover's distance 
def emd(reference_img, uncategorized_img, EMD_cutoff):
    # convert image matrices into numpy arrays
    arr1 = np.array(reference_img) # convert matrix into a numpy array
    arr2 = np.array(uncategorized_img) # convert matrix into a numpy array

    # convert numpy arrays of images into signatures for cv2.EMD
    sig1 = img_to_sig(arr1) 
    sig2 = img_to_sig(arr2)
    print(sig1)
    print(sig2)

    if (sig1 is None) or (sig2 is None):
        # sys.exit("Error: signatures must contain at least one non-zero value")
        return MatchInfo(False, math.inf, True)

    # compute the amount of work required to match smaller weight distribution to larger weight distribution (have all white pixels in the image with fewer white pixels match all of the white pixels in the image with more white pixels)
    work, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)
    print("Work: " + str(work)) # output the amount of work required to transform one distribution into the other

    if work > EMD_cutoff:
        print("No match")
        return MatchInfo(False, work, False)
    else:
        print("Match")
        return MatchInfo(True, work, False)


# takes in two image histograms, a histogram cutoff score, and a histogram comparison method; returns whether the shapes in the two images match up (whether the two images have around the same histogram) and the match score as a MatchInfo object
def color_match(reference_hist, uncategorized_hist, histogram_cutoff, comparison_method):
    # compute whether the two images have similar histograms
    # if they do not, return False
    match_score = cv2.compareHist(reference_hist, uncategorized_hist, comparison_method) # compute the histogram match score between the reference image histogram and the uncategorized image histogram
    print("Histogram match result: " + str(match_score))
    if match_score < histogram_cutoff:
        return MatchInfo(False, match_score, False) # since the histograms of the two images differing significantly, they can't be the same shape
    else:
        return MatchInfo(True, match_score, False) # since the histograms of the two images are around the same, say that they're the same shape


def emd_shape_match(reference_img, uncategorized_img, EMD_cutoff):
    # compute EMD scores
    # if either EMD score is lower than a cutoff score, return True (doesn't require much work to transform one distribution into the other since the shapes are very similar); else return False
    # compute EMD between reference image and uncategorized image
    reference_uncategorized_match = emd(reference_img, uncategorized_img, EMD_cutoff)
    # print("EMD result between reference image and uncategorized image: " + str(reference_uncategorized_match))
    if reference_uncategorized_match.match:
        return reference_uncategorized_match # not a big difference between the reference image and uncategorized image, so consider them as having the same shape

    # compute EMD between reference image and the uncategorized image reflected across the y-axis
    reference_reflected_uncategorized_match = emd(reference_img, np.fliplr(uncategorized_img), EMD_cutoff)
    # print("EMD result between reference image and the uncategorized image reflected across the y-axis: " + str(reference_reflected_uncategorized_match))
    if reference_reflected_uncategorized_match.match:
        return reference_reflected_uncategorized_match # not a big difference between the reference image and the uncategorized image reflected across the y-axis, so consider them as having the same shape

    # determine lowest EMD score
    lowest_EMD = reference_reflected_uncategorized_match.score
    if reference_uncategorized_match.score < reference_reflected_uncategorized_match.score:
        lowest_EMD = reference_uncategorized_match.score
    return MatchInfo(False, lowest_EMD, False)  # EMD scores are too high, so classify the two shapes as not the same; return the lowest EMD score between the 2 images


def main():
    # Read in command line arguments
    num_args = len(sys.argv)

    if num_args < 4:
        # Error
        print("ERROR: Program usage: python3 combination.py (-all, -one, or -specific) (image folder name) [specific flag parameters] (x-resolution) (y-resolution)")
        return -1

    # Determine program mode (-all, -one, or -specific)
    # -all categorizes all images in a folder; default -all
    # -specific determines whether two images have the same fractal shape
    # -one finds all images that have the same shape as the reference image
    program_mode = sys.argv[1] # the program mode: either -all or -specific
    folder_name = sys.argv[2] # the name of the folder within the program's directory in which all images to categorize are stored
    histogram_cutoff = 0.9999 # histogram difference cutoff
    comparison_method = cv2.HISTCMP_CORREL # histogram comparison method
    EMD_cutoff = 1.75 # EMD work cutoff
    white_threshold = 0.005 # white pixel:total pixel ratio cutoff for shapeless images

    hist_output_path = r'hist_initial_output' # output folder path
    if not os.path.exists(hist_output_path):
        os.makedirs(hist_output_path)
    EMD_output_path = r'EMD_final_output' # final output folder path
    if not os.path.exists(EMD_output_path):
        os.makedirs(EMD_output_path)
    info_file = open("category_info.txt", 'w+') # create a text file to store categorization information

    # save the command
    info_file.write("Running: ")
    for i in range(num_args):
        info_file.write(sys.argv[i] + " ")

    info_file.write("\nwhite_threshold: " + str(white_threshold) + "\n") # remember the white_threshold value
    info_file.write("histogram_cutoff: " + str(histogram_cutoff) + "\n") # remember the histogram_cutoff threshold value
    info_file.write("comparison_method: cv2.HISTCMP_CORREL\n") # remember the comparison method for cv2.compareHist()
    info_file.write("EMD_cutoff: " + str(EMD_cutoff) + "\n") # remember the EMD_cutoff threshold value

    start_time = datetime.datetime.now() # get program start time to later compute program total runtime
    

    if program_mode == "-specific": # compare two specific images using just histogram comparison
        # Program usage: python3 combination.py -specific (image folder name) (image 1 number) (image 2 number) (x-resolution) (y-resolution)
        if num_args != 7:
            # Error
            print("ERROR: Program usage: python3 combination.py (program mode) (image folder name) (image 1 number) (image 2 number) (x-resolution) (y-resolution)")
            return -1

        reference_image = sys.argv[3] # the file number of the image in the folder to use as the reference image
        uncategorized_image = sys.argv[4] # the file number of the image in the folder to use as the uncategorized image
        x_res = int(sys.argv[5]) # width of image in pixels for resizing
        y_res = int(sys.argv[6]) # height of image in pixels for resizing

        # read in image files to compare
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

        reference_hist = cv2.calcHist([reference_img], [0], None, [256], [0, 256]) # compute histogram of the reference image
        uncategorized_hist = cv2.calcHist([uncategorized_img], [0], None, [256], [0, 256]) # compute histogram of the uncategorized image

        # write out resized images for reference purposes
        reference_img_output_fp = 'hist_initial_output/frame.' + '{:0>6}'.format(reference_image) + '.png' # output reference image filepath
        uncategorized_img_output_fp = 'hist_initial_output/frame.' + '{:0>6}'.format(uncategorized_image) + '.png' # output uncategorized image filepath
        cv2.imwrite(reference_img_output_fp, reference_img)
        cv2.imwrite(uncategorized_img_output_fp, uncategorized_img)

        initial_match = color_match(reference_hist, uncategorized_hist, histogram_cutoff, comparison_method).match # compute whether the two images have similar histograms
        print("Same shape results: " + str(initial_match)) # print out whether the shapes match
        

    elif program_mode == "-one": # create a category with all images that have similar histograms as a specific reference image
        # Program usage: python3 combination.py -one (image folder name) (image to use as reference) (x-resolution) (y-resolution)
        if num_args != 6:
            # Error
            print("ERROR: Program usage: python3 combination.py -one (image folder name) (image to use as reference) (x-resolution) (y-resolution)")
            return -1
        x_res = int(sys.argv[4]) # width of image in pixels for resizing
        y_res = int(sys.argv[5]) # height of image in pixels for resizing
        print("x_res: " + str(x_res) + ", y_res: " + str(y_res))

        # read in reference image file
        reference_image = sys.argv[3] # the file number of the image in the folder to use as the reference image
        reference_img_fp = folder_name + '/frame.' + '{:0>6}'.format(reference_image) + '.ppm' # reference image filepath
        assert os.path.isfile(reference_img_fp), 'file \'{0}\' does not exist'.format(reference_img_fp) # make sure reference image exists at the filepath specified
        reference_img = cv2.resize(cv2.imread(reference_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read reference image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
        reference_hist = cv2.calcHist([reference_img], [0], None, [256], [0, 256]) # compute histogram of the reference image

        # create a new category with the reference image
        new_category_path = r'hist_initial_output/category.sample' # output category folder path
        if not os.path.exists(new_category_path):
            os.makedirs(new_category_path)
        reference_img_output_fp = new_category_path + '/' + 'frame.{:0>6}'.format(reference_image) + '.png'
        cv2.imwrite(reference_img_output_fp, reference_img) 

        directory = r'{}'.format(folder_name) # read this folder for the images to look through to find the same shapes
        for filename in os.listdir(directory):
            if filename.endswith(".ppm") and filename != ('frame.{:0>6}'.format(reference_image) + '.ppm'):
                print(filename)
                # read in uncategorized image file
                uncategorized_img_fp = folder_name + '/' + filename # uncategorized image filepath
                assert os.path.isfile(uncategorized_img_fp), 'file \'{0}\' does not exist'.format(uncategorized_img_fp) # make sure the uncategorized image exists at the filepath specified
                uncategorized_img = cv2.resize(cv2.imread(uncategorized_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read uncategorized image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
                uncategorized_hist = cv2.calcHist([uncategorized_img], [0], None, [256], [0, 256]) # compute histogram of the uncategorized image

                # compare the shapes in the uncategorized image and the reference image
                if (color_match(reference_hist, uncategorized_hist, histogram_cutoff, comparison_method).match):
                    print(str(reference_image) + " & " + str(filename) + " match")
                    # the shapes are the same, so categorize the uncategorized image in the same group as the reference image
                    # make a copy of the uncategorized image file to put in the shape category folder
                    categorized_img_output_fp = new_category_path + '/' + filename[:-3] + 'png'
                    cv2.imwrite(categorized_img_output_fp, uncategorized_img) 

    else: # categorize all images using histogram comparison for initial groupings and EMD within these initial groupings to create final shape categories
        # Program usage: python3 combination.py -all (image folder name) (x-resolution) (y-resolution)
        if num_args != 5:
            # Error
            print("ERROR: Program usage: python3 combination.py -all (image folder name) (x-resolution) (y-resolution)")
            return -1
        x_res = int(sys.argv[3]) # width of image in pixels for resizing
        y_res = int(sys.argv[4]) # height of image in pixels for resizing
        print("x_res: " + str(x_res) + ", y_res: " + str(y_res))

        shapeless_category_path = r'EMD_final_output/category.' + '{:0>3}'.format(0) # output category folder path for shapeless images
        if not os.path.exists(shapeless_category_path):
            os.makedirs(shapeless_category_path)

        images = {} # create a hashtable storing the categorization information of every image in the folder
        # key is the frame number and value is a dictionary {"filename": filename, "histogram": the image histogram, "hist_categorized": boolean True/False for whether the image has been categorized according to histogram matching, "EMD_categorized": boolean True/False for whether the image has been categorized according to EMD, "hist_category_num": shape category number in initial histogram groupings, "error": boolean True/False for whether there are errors in categorization}
        # iterate through every .ppm frame in the folder 
        directory = r'{}'.format(folder_name) # read this folder for the images to categorize
        num_frames = 0 # keep track of the number of frames in the folder to categorize
        num_hist_categories = 0 # indices for initial categories, starting at 0
        histogram_categories = {} # hold categories created using histogram comparison; key is the category number and value is an array of all image filenames in that shape category
        for filename in os.listdir(directory):
            if filename.endswith(".ppm"):
                print(filename)

                # read in image file
                img_fp = folder_name + '/' + filename # reference image filepath
                assert os.path.isfile(img_fp), 'file \'{0}\' does not exist'.format(img_fp) # make sure image exists at the filepath specified
                img = cv2.resize(cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
               
                # print image sizes if they are successfully read
                if img is not None:
                    print('reference_img.size: ', img.shape)
                else:
                    print('imread({0}) -> None'.format(img_fp))

                file_info = {} # dictionary for every image to store the image's categorization info
                file_info["filename"] = filename
                file_info["hist_categorized"] = False # initialize "hist_categorized" to false
                file_info["EMD_categorized"] = False # initialize "EMD_categorized" to false
                file_info["hist_category_num"] = -1 # initialize to -1 when uncategorized using histogram
                file_info["EMD_category_num"] = -1 # initialize to -1 when uncategorized using EMD
                file_info["histogram"] =  cv2.calcHist([img], [0], None, [256], [0, 256]) # compute the image histogram
                file_info["error"] = False # initialize to no errors in categorization

                # remove images with super few white pixels from categorization consideration and place them all in category.000
                # shape_exists
                if (shape_exists(img, white_threshold) == False):
                    # copy shapeless image into shapeless category
                    categorized_output_fp = shapeless_category_path + '/' + filename[-16:-4] + '.png'
                    cv2.imwrite(categorized_output_fp, img) 
                    file_info["hist_categorized"] = True # categorized in shapeless category
                    file_info["EMD_categorized"] = True # categorized in shapeless category
                    file_info["hist_category_num"] = 0 # in shapeless category 
                    file_info["EMD_category_num"] = 0 # in shapeless category
                    
                images[num_frames] = file_info # save in overall images dictionary
                num_frames += 1 # increment total number of frames in the folder to categorize
            else:
                continue # not a .ppm file
            
        # Shape categorization
        print("num_frames: " + str(num_frames))

        # create initial groupings using histogram comparison
        for i in range(num_frames):
            if images[i]["hist_categorized"] == False: # go to the next uncategorized image and set that as the reference image for a new shape category
                # read in reference image file
                reference_img_fp = folder_name + '/' + images[i]["filename"] # reference image filepath
                assert os.path.isfile(reference_img_fp), 'file \'{0}\' does not exist'.format(reference_img_fp) # make sure reference image exists at the filepath specified
                reference_img = cv2.resize(cv2.imread(reference_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read reference image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
              
                num_hist_categories += 1 # increment the number of shape categories
                # create new shape category with this image as reference, category name = num_hist_categories
                new_category_path = r'hist_initial_output/category.' + '{:0>3}'.format(num_hist_categories) # output category folder path
                if not os.path.exists(new_category_path):
                    os.makedirs(new_category_path)
                
                histogram_categories[str(num_hist_categories)] = [] # array of all image numbers in this category
                histogram_categories[str(num_hist_categories)].append(reference_img_fp) # save reference image filepath
                
                info_file.write("\ncategory." + '{:0>3}'.format(num_hist_categories) + "\n") # save category number in text file
                info_file.write("Reference image: " + images[i]["filename"] + "\n") # save the reference image for this shape category in text file
                
                # make a copy of the image file
                categorized_ref_output_fp = new_category_path + '/' + 'frame.{:0>6}'.format(i) + '.png' # copy of reference image categorized filepath
                cv2.imwrite(categorized_ref_output_fp, reference_img) 

                images[i]["hist_categorized"] = True # remember that this image has been categorized
                images[i]["hist_category_num"] = num_hist_categories # remember the category this image is a part of

                if (i > num_frames):
                    continue # this is the last frame
                for j in range(i + 1, num_frames): 
                    if images[j]["hist_categorized"] == False:
                        # read in uncategorized image file
                        uncategorized_img_fp = folder_name + '/' + images[j]["filename"] # uncategorized image filepath
                        assert os.path.isfile(uncategorized_img_fp), 'file \'{0}\' does not exist'.format(uncategorized_img_fp) # make sure the uncategorized image exists at the filepath specified
                        uncategorized_img = cv2.resize(cv2.imread(uncategorized_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read uncategorized image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel

                        initial_match = color_match(images[i]["histogram"], images[j]["histogram"], histogram_cutoff, comparison_method) # see if the two images have similar histograms
                        # compare the shapes in the two images
                        if (initial_match.match):
                            info_file.write(images[j]["filename"] + ": " + str(initial_match.score) + "\n") # save file number corresponding to the histogram match score
                            print(str(i) + " & " + str(j) + " match")
                            # the shapes have similar histograms, so categorize the two images together
                            # make a copy of the image file
                            categorized_img_output_fp = new_category_path + '/' + 'frame.{:0>6}'.format(j) + '.png'
                            cv2.imwrite(categorized_img_output_fp, uncategorized_img) 

                            histogram_categories[str(num_hist_categories)].append(uncategorized_img_fp) # save image filepath for this new image added to the category

                            images[j]["hist_categorized"] = True # remember that this image has been categorized
                            images[j]["hist_category_num"] = num_hist_categories # save the category number the image was categorized in


        # create final shape categories using EMD
        num_final_categories = 0

        for category, image_list in histogram_categories.items():
            for i in range(len(image_list)):
                image = image_list[i] # get the filepath of an image in this shape category
                ref_digits = image[-10:-4] # get the 6 digits in the filename
                img_num = int(ref_digits[:-1].lstrip('0') + ref_digits[-1]) # get the image number, remove leading 0s
                print("IMAGE: " + str(image))
                print("img_num: " + str(img_num))
                if (images[img_num]["EMD_categorized"] == False): # go to the next uncategorized image in this shape category and set that as the reference image for a new shape category
                    print("reading in")
                    # read in reference image file
                    reference_img_fp = image # reference image filepath
                    assert os.path.isfile(reference_img_fp), 'file \'{0}\' does not exist'.format(reference_img_fp) # make sure reference image exists at the filepath specified
                    reference_img = cv2.resize(cv2.imread(reference_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read reference image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
                
                    num_final_categories += 1 # increment the number of shape categories
                    # create new shape category with this image as reference, category name = num_final_categories
                    new_category_path = r'EMD_final_output/category.' + '{:0>3}'.format(num_final_categories) # output category folder path
                    if not os.path.exists(new_category_path):
                        os.makedirs(new_category_path)
                    
                    info_file.write("\ncategory." + '{:0>3}'.format(num_final_categories) + "\n") # save category number in text file
                    info_file.write("Reference image: " + image[-16:] + "\n") # save the reference image for this shape category in text file
                    
                    # make a copy of the image file
                    categorized_ref_output_fp = new_category_path + '/' + image[-16:-4] + '.png' # copy of reference image categorized filepath
                    cv2.imwrite(categorized_ref_output_fp, reference_img) 

                    images[img_num]["EMD_categorized"] = True # remember that this image has been categorized
                    images[img_num]["EMD_category_num"] = num_final_categories # remember the category this image is a part of
            
                    for j in range(i, len(image_list)): 
                        # read in uncategorized image file
                        uncategorized_img_fp = image_list[j] # uncategorized image filepath
                        uncategorized_digits = uncategorized_img_fp[-10:-4] # get the 6 digits in the filename
                        uncategorized_img_num = int(uncategorized_digits[:-1].lstrip('0') + uncategorized_digits[-1]) # get the image number, remove leading 0s
                        print("here: " + str(images[uncategorized_img_num]["EMD_categorized"]))
                        if (images[uncategorized_img_num]["EMD_categorized"] == False): # go to the next uncategorized image in this shape category
                            print("uncategorized")
                            assert os.path.isfile(uncategorized_img_fp), 'file \'{0}\' does not exist'.format(uncategorized_img_fp) # make sure the uncategorized image exists at the filepath specified
                            uncategorized_img = cv2.resize(cv2.imread(uncategorized_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read uncategorized image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel

                            shape_match = emd_shape_match(reference_img, uncategorized_img, EMD_cutoff) # see if the two images have low EMD score and should actually be in a group together
                            if (shape_match.error):
                                images[uncategorized_img_num]["error"] = True
                                images[uncategorized_img_num]["EMD_categorized"] = True # remember that this image has been categorized
                                continue
                            # compare the shapes in the two images
                            if (shape_match.match):
                                info_file.write(image_list[j]+ ": " + str(shape_match.score) + "\n") # save file number corresponding to the EMD score
                                print(str(i) + " & " + str(j) + " match")
                                # the shapes are the same, so categorize the two images together
                                # make a copy of the image file
                                categorized_img_output_fp = new_category_path + '/' + uncategorized_img_fp[-16:-4] + '.png'
                                cv2.imwrite(categorized_img_output_fp, uncategorized_img) 

                                images[uncategorized_img_num]["EMD_categorized"] = True # remember that this image has been categorized
                                images[uncategorized_img_num]["EMD_category_num"] = num_final_categories # save the category number the image was categorized in

        info_file.write("\nHistogram category results: \n") # spacer and header for summary info        
        info_file.write(str(histogram_categories)) 

        info_file.write("\nimages: \n") # spacer and header for summary info        
        info_file.write(str(images))
    
    end_time = datetime.datetime.now() # get program end time to compute program total runtime
    run_time = end_time - start_time # compute program total runtime

    info_file.write("\nRuntime: \n") # spacer and header for runtime info     
    info_file.write("start_time: " + str(start_time) + "\n")
    info_file.write("end_time: " + str(end_time) + "\n")
    info_file.write("run_time: " + str(run_time) + "\n")

    info_file.close() # close the text file
                

if __name__ == "__main__":
    main()