import skimage, theano, os
import numpy as np
from os.path import isfile, join
from dml.tools.preprocessors import BaseProcessor
from dml.excepts import *
from dml.tools.dataflow import ImageDataFlow

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
	datas = ImageDataFlow(preprocess, rescale)
	datas.addSource(directory, label, keepImage)
	return datas.getAll()