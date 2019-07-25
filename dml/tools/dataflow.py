import skimage, theano, os
import numpy as np
from os.path import isfile, join
from dml.tools.preprocessors import *
from dml.tools.datautils import *
from dml.excepts import *

class BaseDataFlow:
	"""
		Return training / testing datas from a directory
		Format: [<inputs>, <outputs>], with <input> shape: (nbLayers, nbEntries, <input_shape>)
		Same shape for output
	"""
	def getSize(self):
		return 0

	def getDatas(self, ids):
		return []

	def getAll(self):
		return []

class DirectDataFlow(BaseDataFlow):
	def __init__(self, datas):
		self.datas = datas
		# self.datas = [[np.asarray(t, dtype=theano.config.floatX) for t in l] for l in datas]

	def getSize(self):
		return len(self.datas[0][0])

	def getDatas(self, ids):
		return [ [ l[ids] for l in io ] for io in self.datas ]

	def getAll(self):
		return self.datas

class ImageDataFlow(BaseDataFlow):
	def __init__(self, preprocess=None, rescale=1/256, augment=None, keepGray=True):
		self.imagesNames = []
		self.imagesPaths = []
		self.imagesLabel = []
		self.preprocess = preprocess
		self.rescale = rescale # Rescale colors
		self.augment = augment
		self.nbAugmentFilters = 1 if augment == None else augment.nbFilters()
		self.keepGray = keepGray # Do not convert to RGB.

	@staticmethod
	def isolateChannels(image):
		""" convert shape (h, w, channels) to (channels, h, w) """
		if isImgGrayscale(image):
			return image # If grayscale
		nbChan = len(image[0][0])
		return np.asarray([
			image[:,:,chan] for chan in range(nbChan)
		], dtype=theano.config.floatX)

	def addImageFromSource(self, fName, path, label):
		for _ in range(self.nbAugmentFilters):
			self.imagesNames.append(fName)
			self.imagesPaths.append(path)
			self.imagesLabel.append(label)

	def addSource(self, directory, label, keepImage=None):
		if isinstance(keepImage, str):
			ext = keepImage
			keepImage = lambda x : x[-len(ext):] == ext

		for f in os.listdir(directory):
			if isfile(join(directory, f)) and (not keepImage or keepImage(f)):
				self.addImageFromSource(f, join(directory, f), label)

	def readImage(self, imgId):
		img = np.asarray(
			skimage.io.imread(self.imagesPaths[imgId]),
			dtype=theano.config.floatX
		) * self.rescale

		if not self.keepGray:
			img = ImagePreprocess.toRGBImage(img)
		if callable(self.preprocess):
			img = self.preprocess(img)
		elif isinstance(self.preprocess, BaseProcessor):
			img = self.preprocess.process(img)

		if self.augment:
			img = self.augment.apply(img, imgId%self.nbAugmentFilters)

		return img

	def getDatas(self, ids):
		images = [self.isolateChannels(self.readImage(img)) for img in ids]

		for img in images:
			if img.shape != images[0].shape:
				raise ShapeError("All images must have the same size")

		labels = [
			self.imagesLabel[img](self.imagesNames[img]) if callable(self.imagesLabel[img])
			else self.imagesLabel[img]
			for img in ids
		]

		return [
			[np.asarray(images, dtype=theano.config.floatX)],
			[np.asarray(labels, dtype=theano.config.floatX)],
		]

	def getAll(self):
		return self.getDatas(np.arange(self.getSize()))

	def getSize(self):
		return len(self.imagesNames)