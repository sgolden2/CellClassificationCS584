import numpy as np
import cv2 
import sys
import os

WINDOW_TITLE = "Cell Splitter"

imageDIR = "./hpa-single-cell-image-classification/train"
imageID = "000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0"
imageTags = ['_blue', '_red', '_yellow', '_green']
fileEXT = '.png'

CANNY_HIGH_THRESHOLD = 254
CANNY_KERNEL_SIZE = 3
BINARIZATION_THRESHOLD = 100
CANNY_LOW_THRESHOLD = BINARIZATION_THRESHOLD + 1
MAX_CLUSTER_COUNT = 19

def make_bw_binary(image):
	image[image > BINARIZATION_THRESHOLD] = 255
	image[image <= BINARIZATION_THRESHOLD] = 0
	return image

def make_binary(image):
	image[image > 0] = 1
	return image

def detect_edges(image):
	global CANNY_LOW_THRESHOLD
	global CANNY_HIGH_THRESHOLD
	global CANNY_KERNEL_SIZE

	edges = cv2.Canny(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, CANNY_KERNEL_SIZE)
	return edges

def downsample_nosmooth(image):
	return cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

def smooth(image):
	return cv2.GaussianBlur(image,(5,5),0)

# expects grayscale, b&w image (will binarize everything above 0 to 1)
def get_clusters(image):
	global MAX_CLUSTER_COUNT

	image = make_binary(image)
	points = np.float32(np.column_stack(np.flip(np.where(image==1))))
	
	best_cluster_points = None
	best_cluster_error = None
	best_cluster_k = None

	for k in range(1, MAX_CLUSTER_COUNT):
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		err,labels,centers = cv2.kmeans(points,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

		if k==1:
			best_cluster_error = err
			best_cluster_points = labels
			best_cluster_k = k
		else:
			if err < best_cluster_error:
				best_cluster_error = err
				best_cluster_points = labels
				best_cluster_k = k

	clusters = []
	for i in range(best_cluster_k):
		clusters.append(points[best_cluster_points.ravel()==i].astype(np.uint8))

	return points,best_cluster_points,clusters

def visualize_clusters(clusters, points, dimensions):
	red = np.uint8([[[0,0,255]]])
	blue = np.uint8([[[255,0,0]]])
	green = np.uint8([[[0,255,0]]])
	colors = (red,blue,green)

	image = np.zeros(dimensions+(3,)).astype(np.uint8)
	#image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	clusters = clusters.astype(np.uint16)
	points = points.astype(np.uint16)

	for i in range(points.shape[0]):
		image[points[i][0],points[i][1]] = colors[clusters[i][0] % len(colors)]

	return image

filenames = [os.path.join(imageDIR, imageID + tag + fileEXT) for tag in imageTags]

images = []

for i in range(len(filenames)):
	images.append(cv2.cvtColor(cv2.imread(filenames[i]), cv2.COLOR_BGR2GRAY))

images = np.array(images)

summed = np.clip(np.sum(images, axis=0), 1, 254).astype(np.uint8)
smoothed = make_bw_binary(smooth(summed))
edges = downsample_nosmooth(detect_edges(smoothed))

todisplay1 = downsample_nosmooth(downsample_nosmooth(summed))
todisplay2 = downsample_nosmooth(downsample_nosmooth(smoothed))
todisplay3 = downsample_nosmooth(edges)

points,cluster_points,clusters = get_clusters(smoothed)

displayable = np.concatenate((todisplay1,todisplay2,todisplay3), axis=1)
cv2.imshow(WINDOW_TITLE, displayable)
cv2.waitKey(0)

cv2.imshow(WINDOW_TITLE, downsample_nosmooth(visualize_clusters(cluster_points, points, smoothed.shape)))
cv2.waitKey(0)
