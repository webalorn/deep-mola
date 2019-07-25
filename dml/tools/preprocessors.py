from skimage import transform, util, color
from dml.tools.datautils import isImgGrayscale

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

	@staticmethod
	def toRGBImage(image):
		if len(image.shape) == 3:
			return image
		return color.grey2rgb(image)

	def process(self, image):
		if self.newShape:
			if self.keepShape:
				h, w = image.shape[0], image.shape[1]
				scale = min(self.newShape[0]/h, self.newShape[1]/w)
				padH = round((self.newShape[0] / scale - h) / 2)
				padW = round((self.newShape[1] / scale - w) / 2)

				padShape = ((padH, padH), (padW, padW))
				if not isImgGrayscale(image):
					padShape += ((0, 0),)
				image = util.pad(image, padShape, 'constant')

			image = transform.resize(image, self.newShape, mode='symmetric', preserve_range=True)
			if self.grayscale and not isImgGrayscale(image):
				image = color.rgb2gray(image)
			elif not self.grayscale:
				image = ImagePreprocess.toRGBImage(image)

		return image