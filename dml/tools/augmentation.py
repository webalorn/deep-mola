import skimage
from dml.tools.datafilers import *

class DataAugment:
	def __init__(self):
		self.filters = [DataFilter()]

	def nbFilters(self):
		"""
			Returns the number of filters, including filter leaving the same image
		"""
		return len(self.filters)

	def apply(self, image, iFilter):
		return self.filters[iFilter].apply(image)

	def addFilters(self, filtersObjs, foreach=False):
		if not isinstance(filtersObjs, list):
			filtersObjs = [filtersObjs]
		if foreach:
			newFilters = []
			for filtObj in filtersObjs:
				for f in self.filters:
					newFilters.append(filterObj.withPrevious(f))
			self.filters += newFilters
		else:
			for f in filtersObjs:
				self.filters.append(f)

	def addRotations(self, angles=[90, 180, 270], foreach=False):
		self.addFilters([RotationFilter(alpha) for alpha in angles], foreach)

	def addMirror(self, foreach=False):
		self.addFilters(MirrorFilter(), foreach)