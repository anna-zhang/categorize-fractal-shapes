import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil
import sys # process command line arguments

class MatchInfo:
  def __init__(self, match, score):
    self.match = match # True if the two shapes match, False if the two shapes do not match
    self.score = score # match score between the two shapes

# takes in two image histograms, a histogram cutoff score, and a histogram comparison method; returns whether the shapes in the two images match up (whether the two images have around the same histogram) and the match score as a MatchInfo object
def same_shape(reference_hist, uncategorized_hist, histogram_cutoff, comparison_method):
    # compute whether the two images have similar histograms
    # if they do not, return False
    match_score = cv2.compareHist(reference_hist, uncategorized_hist, comparison_method) # compute the histogram match score between the reference image histogram and the uncategorized image histogram
    print("Histogram match result: " + str(match_score))
    if match_score < histogram_cutoff:
        return MatchInfo(False, match_score) # since the histograms of the two images differing significantly, they can't be the same shape
    else:
        return MatchInfo(True, match_score) # since the histograms of the two images are around the same, say that they're the same shape

def main():
    # Read in command line arguments
    num_args = len(sys.argv)

    if num_args < 4:
        # Error
        print("ERROR: Program usage: python3 histogram-match.py (-all, -one, or -specific) (image folder name) [specific flag parameters] (x-resolution) (y-resolution)")
        return -1

    # Determine program mode (-all, -one, or -specific)
    # -all categorizes all images in a folder; default -all
    # -specific determines whether two images have the same fractal shape
    # -one finds all images that have the same shape as the reference image
    program_mode = sys.argv[1] # the program mode: either -all or -specific
    folder_name = sys.argv[2] # the name of the folder within the program's directory in which all images to categorize are stored
    histogram_cutoff = 0.99999 # histogram difference cutoff
    comparison_method = cv2.HISTCMP_CORREL # histogram comparison method

    output_path = r'output' # output folder path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    info_file = open("output/category_info.txt", 'w+') # create a text file to store categorization information

    # save the command
    info_file.write("Running: ")
    for i in range(num_args):
        info_file.write(sys.argv[i] + " ")
    info_file.write("\nhistogram_cutoff: " + str(histogram_cutoff) + "\n") # remember the histogram_cutoff threshold value
    info_file.write("\ncomparison_method: " + str(comparison_method) + "\n") # remember the comparison method for cv2.compareHist()

    if program_mode == "-specific": # compare two specific images
        # Program usage: python3 histogram-match.py -specific (image folder name) (image 1 number) (image 2 number) (x-resolution) (y-resolution)
        if num_args != 7:
            # Error
            print("ERROR: Program usage: python3 histogram-match.py (program mode) (image folder name) (image 1 number) (image 2 number) (x-resolution) (y-resolution)")
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
        reference_img_output_fp = 'output/frame.' + '{:0>6}'.format(reference_image) + '.png' # output reference image filepath
        uncategorized_img_output_fp = 'output/frame.' + '{:0>6}'.format(uncategorized_image) + '.png' # output uncategorized image filepath
        cv2.imwrite(reference_img_output_fp, reference_img)
        cv2.imwrite(uncategorized_img_output_fp, uncategorized_img)

        shape_match = same_shape(reference_hist, uncategorized_hist, histogram_cutoff, comparison_method).match # compute whether the two images have similar histograms
        print("Same shape results: " + str(shape_match)) # print out whether the shapes match

    elif program_mode == "-one": 
        # Program usage: python3 histogram-match.py -one (image folder name) (image to use as reference) (x-resolution) (y-resolution)
        if num_args != 6:
            # Error
            print("ERROR: Program usage: python3 histogram-match.py -one (image folder name) (image to use as reference) (x-resolution) (y-resolution)")
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
        new_category_path = r'output/category.sample' # output category folder path
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
                if (same_shape(reference_hist, uncategorized_hist, histogram_cutoff, comparison_method).match):
                    print(str(reference_image) + " & " + str(filename) + " match")
                    # the shapes are the same, so categorize the uncategorized image in the same group as the reference image
                    # make a copy of the uncategorized image file to put in the shape category folder
                    categorized_img_output_fp = new_category_path + '/' + filename[:-3] + 'png'
                    cv2.imwrite(categorized_img_output_fp, uncategorized_img) 

    else: # categorize all images
        # Program usage: python3 histogram-match.py -all (image folder name) (x-resolution) (y-resolution)
        if num_args != 5:
            # Error
            print("ERROR: Program usage: python3 histogram-match.py -all (image folder name) (x-resolution) (y-resolution)")
            return -1
        x_res = int(sys.argv[3]) # width of image in pixels for resizing
        y_res = int(sys.argv[4]) # height of image in pixels for resizing
        print("x_res: " + str(x_res) + ", y_res: " + str(y_res))

        images = {}
        # create a hashtable storing the categorization information of every image in the folder
        # key is the frame number and value is a dictionary {"filename": filename, "histogram": the image histogram, "categorized": boolean True/False, "category_num": shape category number}
        # iterate through every .ppm frame in the folder 
        directory = r'{}'.format(folder_name) # read this folder for the images to categorize
        num_frames = 0 # keep track of the number of frames in the folder to categorize
        num_categories = 0 # indices for categories, starting at 0
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
                file_info["categorized"] = False # initialize "categorized" to false
                file_info["category_num"] = -1 # initialize to -1 when uncategorized
                file_info["histogram"] =  cv2.calcHist([img], [0], None, [256], [0, 256]) # compute the image histogram
                    
                images[num_frames] = file_info # save in overall images dictionary
                num_frames += 1 # increment total number of frames in the folder to categorize
            else:
                continue # not a .ppm file
            
        # Shape categorization
        print("num_frames: " + str(num_frames))
        for i in range(num_frames):
            if images[i]["categorized"] == False: # go to the next uncategorized image and set that as the reference image for a new shape category
                # read in reference image file
                reference_img_fp = folder_name + '/' + images[i]["filename"] # reference image filepath
                assert os.path.isfile(reference_img_fp), 'file \'{0}\' does not exist'.format(reference_img_fp) # make sure reference image exists at the filepath specified
                reference_img = cv2.resize(cv2.imread(reference_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read reference image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel
              
                num_categories += 1 # increment the number of shape categories
                # create new shape category with this image as reference, category name = num_categories
                new_category_path = r'output/category.' + '{:0>3}'.format(num_categories) # output category folder path
                if not os.path.exists(new_category_path):
                    os.makedirs(new_category_path)
                
                info_file.write("\ncategory." + '{:0>3}'.format(num_categories) + "\n") # save category number in text file
                
                # make a copy of the image file
                categorized_ref_output_fp = new_category_path + '/' + 'frame.{:0>6}'.format(i) + '.png' # copy of reference image categorized filepath
                cv2.imwrite(categorized_ref_output_fp, reference_img) 

                images[i]["categorized"] = True # remember that this image has been categorized
                images[i]["category_num"] = num_categories # remember the category this image is a part of
        
                for j in range(i, num_frames): 
                    # read in uncategorized image file
                    uncategorized_img_fp = folder_name + '/' + images[j]["filename"] # uncategorized image filepath
                    assert os.path.isfile(uncategorized_img_fp), 'file \'{0}\' does not exist'.format(uncategorized_img_fp) # make sure the uncategorized image exists at the filepath specified
                    uncategorized_img = cv2.resize(cv2.imread(uncategorized_img_fp, cv2.IMREAD_GRAYSCALE), (x_res, y_res)) # read uncategorized image as an array of grayscale values so only getting one value per pixel instead of values for three color channels per pixel

                    shape_match = same_shape(images[i]["histogram"], images[j]["histogram"], histogram_cutoff, comparison_method) # see if the two images have similar histograms
                    # compare the shapes in the two images
                    if (shape_match.match):
                        info_file.write(images[j]["filename"] + ": " + str(shape_match.score) + "\n") # save file number corresponding to the histogram match score
                        print(str(i) + " & " + str(j) + " match")
                        # the shapes are the same, so categorize the two images together
                        # make a copy of the image file
                        categorized_img_output_fp = new_category_path + '/' + 'frame.{:0>6}'.format(j) + '.png'
                        cv2.imwrite(categorized_img_output_fp, uncategorized_img) 

                        images[j]["categorized"] = True # remember that this image has been categorized
                        images[j]["category_num"] = num_categories # save the category number the image was categorized in
        
        info_file.write("\n Categorization results: \n") # spacer and header for summary info        
        info_file.write(str(images))  
    info_file.close() # close the text file
                

if __name__ == "__main__":
    main()