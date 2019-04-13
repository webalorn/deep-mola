import skimage, theano, os
import numpy as np
from os.path import isfile, join
from dml.tools.preprocessors import BaseProcessor
from dml.excepts import *

def isolateChannels(image):
	""" convert shape (h, w, channels) to (channels, h, w) """
	nbChan = len(image[0][0])
	return np.asarray([
		image[:,:,chan] for chan in range(nbChan)
	], dtype=theano.config.floatX)

def readImagesFrom(directory, label, keepImage=None, preprocess=None, rescale=1/256):
	"""
		Directory is the path to a directory to read images from.
		label is a constant or a function called to label examples.
		It must take the filename as parameter.
		keepImage may be a function (or None) called to know if the image must be kept.
		If keepImage is a string, the file must have the given extension
		It must take the filename as parameter
		rescale might be used to have reals between 0 and 1, without needing a preprocessor.

		You can provide a list instead of the arguments to merge multiple sources
	"""

	if isinstance(directory, list):
		ins, outs = [], []

		allArgs = [directory, label, keepImage]
		for i, arg in enumerate(allArgs):
			if not isinstance(arg, list):
				allArgs[i] = [arg] * len(directory)

		for args in zip(*allArgs):
			newIn, newOut = readImagesFrom(*args)

	if isinstance(keepImage, str):
		ext = keepImage
		keepImage = lambda x : x[-len(ext):] == ext

	imgNames = [
		f for f in os.listdir(directory)
		if isfile(join(directory, f)) 
			and (not keepImage or keepImage(f))
	]

	images = [
		np.asarray(
			skimage.io.imread(join(directory, img)),
			dtype=theano.config.floatX
		) * rescale
		for img in imgNames
	]

	if callable(preprocess):
		images = [preprocess(img) for img in images]
	elif isinstance(preprocess, BaseProcessor):
		images = [preprocess.process(img) for img in images]

	for img in images:
		if img.shape != images[0].shape:
			raise ShapeError("All images must have the same size")

	labels = [label(img) for img in imgNames] if callable(label) else [label] * len(imgNames)

	images = [isolateChannels(img) for img in images]

	return [
		np.asarray(images, dtype=theano.config.floatX),
		np.asarray(labels, dtype=theano.config.floatX),
	]