# Using K-Means Clustering for Image Segregation by Hrishi Patel: https://towardsdatascience.com/using-k-means-clustering-for-image-segregation-fd80bea8412d
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil

# read in all images 
input_dir = 'PNGs'
glob_dir = input_dir + '/*.png'
images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)] # resize images to 224x224 so that it matches with input layer size
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images), -1)/255) # image input divided by 255 so input values in range [0, 1]

# extract features
model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3)) # feature extraction with MobileNetV2
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)


k = 60 # number of clusters
kmodel = KMeans(n_clusters = k, random_state=728)
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
shutil.rmtree('output')
for i in range(k):
    os.makedirs("output/cluster" + str(i))
for i in range(len(paths)):
    shutil.copy2(paths[i], "output/cluster" + str(kpredictions[i]))