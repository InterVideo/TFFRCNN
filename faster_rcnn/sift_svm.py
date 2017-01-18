import numpy as np
from scipy import spatial
import cv2
from cv2.xfeatures2d import SIFT_create
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import normalize
from retry import retry
from time import time

from faster_rcnn.data_augmentation import data_augmentation


class RootSIFT(object):
    def __init__(self):
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps, eps=1e-7):
        kps, descs = self.extractor.compute(image, kps)

        if len(kps) == 0:
            return [], None

        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        return kps, descs

    def detectAndCompute(self, image):
        kps = self.extractor.detect(image)
        return self.compute(image, kps)[1]


class DenseRootSIFT(object):
	def __init__(self, window_size=(10, 10)):
		self.window_size = window_size
		self.sift = cv2.xfeatures2d.SIFT_create()

	def detectAndCompute(self, image, step_size=12):
		if self.window_size is None:
			winH, winW = image.shape[:2]
			self.window_size = (winW // 4, winH // 4)

		descriptors = np.array([], dtype=np.float32).reshape(0, 128)
		for crop in self._crop_image(image, step_size, self.window_size):
			crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
			descs = self._detectAndCompute(crop)[1]
			if descs is not None:
				descriptors = np.vstack([descriptors, self._detectAndCompute(crop)[1]])
		return descriptors

	def _detect(self, image):
		return self.sift.detect(image)

	def _compute(self, image, kps, eps=1e-7):
		kps, descs = self.sift.compute(image, kps)

		if len(kps) == 0:
			return [], None

		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		return kps, descs

	def _detectAndCompute(self, image):
		kps = self._detect(image)
		return self._compute(image, kps)

	def _sliding_window(self, image, step_size, window_size):
		for y in xrange(0, image.shape[0], step_size):
			for x in xrange(0, image.shape[1], step_size):
				yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

	def _crop_image(self, image, step_size=12, window_size=(10, 10)):
		crops = []
		winH, winW = window_size
		for (x, y, window) in self._sliding_window(image, step_size=step_size, window_size=(winW, winH)):
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			crops.append(image[y:y+winH, x:x+winW])
		return np.array(crops)


class SiftFeaturesPreparator(object):
	def __init__(self, histogram_size=2048, dense_sift=False, clustering='kmeans'):
		self.X = []
		self.codebook = []
		self.clustering = clustering
		
		if dense_sift:
			self.sift = DenseRootSIFT()
		else:
			self.sift = RootSIFT()

		self.histogram_size = histogram_size

	def fit_transform(self, dataset):
		self._fit(dataset)
		return self.extract_descriptors_and_prepare_for_classification()

	def _fit(self, image_dataset):
		# @param image_dataset - array of images in OpenCV format
		self.X = image_dataset

	def extract_descriptors_and_prepare_for_classification(self):
		return self._get_histograms()

	def _get_histograms(self):
		codebook = self._generate_codebook(clustering=self.clustering)
		histograms = []

		img_count = 0

		for img in self.X:
			img_count += 1
			print 'Creating histogram for image ' + str(img_count) + ' out of ' + str(len(self.X))
			histogram = self._create_histogram(img, self.histogram_size, codebook)
			histograms.append(histogram)
		return np.array(histograms)

	def _create_histogram(self, image, hist_size, codebook):
		histogram = np.zeros(hist_size)
		descriptors = self.sift.detectAndCompute(image)
		tree = spatial.KDTree(codebook)

		for i in xrange(len(descriptors)):
			histogram[tree.query(descriptors[i])[1]] += 1

		return normalize(histogram[:, np.newaxis], axis=0).ravel()

	@retry(IndexError, tries=3)
	def _generate_codebook(self, clustering='kmeans'):
		descriptors = np.array([], dtype=np.float32).reshape(0, 128)
		for image in self.X:
			descs = self.sift.detectAndCompute(image)
			descriptors = np.vstack((descriptors, descs))

		n_clusters = 2048
		if len(descriptors) < 4096:
			n_clusters = len(descriptors) // 2

		if clustering == 'gmm':
			return self._gmm_clustering(descriptors, n_clusters)
		else clustering == 'kmeans':
			return self._kmeans_clustering(descriptors, n_clusters)

	def _kmeans_clustering(self, X, n_clusters, batch_size=128):
		kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
								 n_init=10, max_no_improvement=10)
		kmeans.fit(X)
		return kmeans.cluster_centers_

	def _gmm_clustering(self, X, n_components):
		gmm = GaussianMixture(n_components=n_components)
		gmm.fit(X)
		return gmm.means_


def sliding_window(image, step_size, window_size):
	for y in xrange(0, image.shape[0], step_size):
		for x in xrange(0, image.shape[1], step_size):
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def calculate_intersection_ratio(rect1=(0,0,0,0), rect2=(0,0,0,0)):
	x1, y1, w1, h1 = rect1
	x2, y2, w2, h2 = rect2 
	x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
	y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
	overlap_area = float(x_overlap) * float(y_overlap)
	non_overlap_area = (w1 * h1) + (w2 * h2) - overlap_area
	return overlap_area / non_overlap_area


# @param image - original image in OpenCV format.
# 
# @param cropped_positive_image - cropped image from the original image (@image)
#  that contains the positive training example in OpenCV format. This image should
#  be avoided when generating negative crops from the original image (@image).
# 
# @param cropped_image_point_on_original - the upper left corner
#  point of @cropped_positive_image.
# 
# @params points_only - if set to True, returns the array of (x, y, width, height)
#  tuples for each cropped negative training example. If set to False, returns
#  the array of cropped negative training images in the OpenCV format.
def get_negative_crops(
	image,
	cropped_positive_image,
	cropped_image_point_on_original=(0,0),
	points_only=False):

	winH, winW = cropped_positive_image.shape[:2]

	x2, y2 = cropped_image_point_on_original

	negative_crops = []

	for (x, y, window) in sliding_window(image, step_size=12, window_size=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		intersect_ratio = calculate_intersection_ratio(
			rect1=(x, y, winW, winH),
			rect2=(x2, y2, winW, winH)
		)

		too_large_intersect = True if intersect_ratio > 0.1 else False

		if not too_large_intersect:
			negative_crops.append(
				(x, y, x + winW, y + winH) if points_only else image[y:y+winH, x:x+winW]
			)

	return np.array(negative_crops)


@retry(IndexError, tries=3)
def generate_codebook(image, detectAndCompute=SIFT_create().detectAndCompute):
	descriptors = detectAndCompute(image, window_size=None)
	kmeans = MiniBatchKMeans(n_clusters=2048, batch_size=128,
							 n_init=10, max_no_improvement=10)
	kmeans.fit(descriptors)
	codebook = kmeans.cluster_centers_[:]
	return codebook


def create_histogram(image, hist_size=2048, codebook=[], detectAndCompute=SIFT_create().detectAndCompute):
	histogram = np.zeros(hist_size)
	descriptors = detectAndCompute(image, window_size=None)
	tree = spatial.KDTree(codebook)

	for i in xrange(len(descriptors)):
		histogram[tree.query(descriptors[i])[1]] += 1

	return normalize(histogram[:, np.newaxis], axis=0).ravel()


def create_dataset_for_svm(original_image, cropped_positive_image, cropped_image_point_on_original):
    negatives = get_negative_crops(
    	original_image, cropped_positive_image, cropped_image_point_on_original
    )
	negatives = data_augmentation(negatives)
	print 'NEGATIVES_LENGTH (sift_svm.py):', len(negatives)
    y = -np.ones(len(negatives))
    X = np.vstack([negatives, [cropped_positive_image]])
    y = np.concatenate((y, [1]))
    return X, y


def train_svm(X, y):
   svm = LinearSVC()
   svm.fit(X, y)
   return svm


# img = cv2.imread('cat.jpg')
# rsz_cat = cv2.imread('rsz_cat.jpg')

# X, y = create_dataset_for_svm(img, rsz_cat, (275, 3))
# print X.shape, y.shape
# print y

# t1 = time()
# X2 = SiftFeaturesPreparator(dense_sift=True).fit_transform(X, rsz_cat)
# t2 = t1 - time()

# t1 = time()
# X1 = SiftFeaturesPreparator(dense_sift=False).fit_transform(X, rsz_cat)
# t3 = t1 - time()

# print t2
# print t3

# print X1.shape
# print X2.shape
# print all(all(a) for a in X1 == X2)

# Step 1. Get the images array from the dataset
# img = cv2.imread('cat.jpg')
# rsz_cat = cv2.imread('rsz_cat.jpg')
# neg_crops = get_negative_crops(img, rsz_cat, (275, 3))

# winH, winW = rsz_cat.shape[:2]
# pos_crop = img[3:3+winH, 275:275+winW]

# images = np.vstack([neg_crops, [pos_crop]])
# np.random.shuffle(images)

# print 'Step 1 done. Images are ready for processing'


# dense_root_sift = DenseRootSIFT()


# Create a codebook for input test image
# test_input_img = cv2.imread('rsz_cat.jpg')

# t0 = time()

# histogramExtractor = DenseRootSiftPreparator()
# print 'extractor created'
# histogramExtractor.fit(images)
# print 'dataset has been fitted'
# hists = histogramExtractor.extract_descriptors_and_prepare_for_classification(test_input_img)
# print 'histograms have been created'
# print np.array(hists).shape

# t1 = time() - t0
# print 'Time: ' + str(t1)

# codebook = generate_codebook(test_input_img, dense_root_sift.detectAndCompute)

# print '------------------CODEBOOK----------------------'
# print codebook
# print '------------------------------------------------'


# histograms = []
# i = 1
# for image in images:
# 	histogram = create_histogram(image, 2048, codebook, dense_root_sift.detectAndCompute)
# 	histograms.append(histogram)

# the "histograms"" array is ready for SVM classification ???



# t1 = time() - t0
# print 'Time: ' + str(t1)


# print root_sift.detectAndCompute(img, window_size=None).shape

# Step 3. Cluster all descriptors using KMeans
# t1 = time()
# kmeans = MiniBatchKMeans(n_clusters=2048, batch_size=128,
#                          n_init=10, max_no_improvement=10)
# kmeans.fit(descriptors)
# t_mini_batch = time() - t1

# print 'MiniBatchKMeans time: ' + str(t_mini_batch)
# print kmeans.labels_.shape
# print kmeans.cluster_centers_.shape

# print 'Step 3 done. Descriptors have been clustered'

# Step 4. Build histograms
# histograms = np.array([])

# for image in images:
# 	hist = np.zeros(2048)

