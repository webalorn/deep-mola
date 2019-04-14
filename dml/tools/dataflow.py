import skimage, theano, os
import numpy as np
from os.path import isfile, join
from dml.tools.preprocessors import BaseProcessor
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
	def __init__(self, preprocess=None, rescale=1/256):
		self.imagesNames = []
		self.imagesPaths = []
		self.imagesLabel = []
		self.preprocess = preprocess
		self.rescale = rescale

	@staticmethod
	def isolateChannels(image):
		""" convert shape (h, w, channels) to (channels, h, w) """
		nbChan = len(image[0][0])
		return np.asarray([
			image[:,:,chan] for chan in range(nbChan)
		], dtype=theano.config.floatX)

	def addSource(self, directory, label, keepImage=None):
		if isinstance(keepImage, str):
			ext = keepImage
			keepImage = lambda x : x[-len(ext):] == ext

		for f in os.listdir(directory):
			if isfile(join(directory, f)) and (not keepImage or keepImage(f)):
				self.imagesNames.append(f)
				self.imagesPaths.append(join(directory, f))
				self.imagesLabel.append(label)

	def readImage(self, imgId):
		img = np.asarray(
			skimage.io.imread(self.imagesPaths[imgId]),
			dtype=theano.config.floatX
		) * self.rescale

		if callable(self.preprocess):
			img = self.preprocess(img)
		elif isinstance(self.preprocess, BaseProcessor):
			img = self.preprocess.process(img)
		return img

	def getDatas(self, ids):
		images = [self.readImage(img) for img in ids]

		for img in images:
			if img.shape != images[0].shape:
				raise ShapeError("All images must have the same size")

		labels = [
			self.imagesLabel[img](self.imagesNames[img]) if callable(self.imagesLabel[img])
			else self.imagesLabel[img]
			for img in ids
		]

		images = [self.isolateChannels(img) for img in images]

		return [
			[np.asarray(images, dtype=theano.config.floatX)],
			[np.asarray(labels, dtype=theano.config.floatX)],
		]

	def getAll(self):
		return self.getDatas(np.arange(self.getSize()))

	def getSize(self):
		return len(self.imagesNames)