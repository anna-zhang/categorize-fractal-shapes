# Earth Mover's Distance (EMD)
This uses Earth Mover's Distance to compute the difference between two images. This involves transforming the images into "signatures", which are ways of representing distributions, and then computing the distance between the two signatures. Signatures are formatted as a matrix with every pixel specified as [point weight (pixel value), x-coordinate, y-coordinate]. The cv2.EMD() function takes in two distributions and returns the distance between the distributions, the lower-bound (the distance between the centers of mass of the two signatures), and a flow matrix (every element i,j in the matrix represents the amount of weight transferred from the ith position in the first signature to the jth position in the second signature). The flow matrix essentially tells you what moved where between the two distributions.

The two images to compare are the images located at the filepaths specified by the reference_img_fp and uncategorized_img_fp variables.

The code follows [an EMD tutorial](https://gist.github.com/svank/6f3c2d776eea882fd271bba1bd2cc16d) by Sam Van Kooten.

# Categorization
Images are first resized to a lower resolution. All images are set to uncategorized. Iterate through all of the images. The next image that has not been categorized is set as the “reference” image for a new shape category that is created. All remaining images that have not been categorized are compared with this reference image. If the two images have similar white pixel to total pixel ratios, Earth Mover's Distance is computed. If the work score to transform one image into the other is lower than a set threshold value, the uncategorized image is added to the reference image's shape category and classified as categorized.
