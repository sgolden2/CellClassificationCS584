import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import random

train_image_path = './hpa-single-cell-image-classification/train'
train_image_filenames = [f for f in listdir(train_image_path) if isfile(join(train_image_path, f)) and 'green' in f]


for i in range(10):
	example = join(train_image_path, random.choice(train_image_filenames))

	print(example)

	img = cv.imread(example,0)
	img = cv.medianBlur(img,5)

	ret,th1 = cv.threshold(img,100,255,cv.THRESH_BINARY)
	th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,21,2)
	th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,21,2)

	titles = ['Original Image', 'Global Thresholding (v = 127)',
	            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img, th1, th2, th3]

	for i in range(4):
	    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])
	plt.show()