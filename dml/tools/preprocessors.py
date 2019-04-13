from skimage import transform, util

class BaseProcessor:
	def process(self, inData):
		return inData


class ImagePreprocess(BaseProcessor):
	"""
		Process images with shape (height, width, channels) as used in skimage
	"""
	def __init__(self, newShape=None, keepShape=True, grayscale=False): # TODO
		self.newShape = newShape # None or (newHeight, newWidth)
		self.keepShape = keepShape # Fill borders with 0
		self.grayscale = grayscale

	def process(self, image):
		if self.newShape:
			if self.keepShape:
				h, w = image.shape[0], image.shape[1]
				scale = min(self.newShape[0]/h, self.newShape[1]/w)
				padH = round((self.newShape[0] / scale - h) / 2)
				padW = round((self.newShape[1] / scale - w) / 2)
				image = util.pad(image, ((padH, padH), (padW, padW), (0, 0)), 'constant')

			image = transform.resize(image, self.newShape, mode='symmetric', preserve_range=True)

		return image