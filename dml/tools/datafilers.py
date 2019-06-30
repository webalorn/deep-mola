import skimage

class DataFilter():
	"""
		Default filter, do nothing
	"""
	def __init__(self, prevFilter=None):
		self.prevFilter = prevFilter

	def withPrevious(self, prevFilter):
		self.prevFilter = prevFilter
		return self

	def _internalApply(self, image):
		return image

	def apply(self, image):
		image = self._internalApply(image)
		if self.prevFilter:
			return self.prevFilter.apply(image)
		return image

class RotationFilter(DataFilter):
	def __init__(self, alpha, prevFilter=None):
		super().__init__(prevFilter)
		self.alpha = alpha

	def _internalApply(self, image):
		return skimage.transform.rotate(image, self.alpha)

class MirrorFilter(DataFilter):
	def _internalApply(self, image):
		return image[:, ::-1]

class GaussianFilter(DataFilter):
	def _internalApply(self, image):
		return skimage.filters.gaussian(image)