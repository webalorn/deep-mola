import numpy as np

class Sigmoid:
	@staticmethod
	def apply(x):
		return 1. / ( 1 + np.exp(-1 * x) )

	@staticmethod
	def derivative(x):
		return Sigmoid.apply(x) * (1 - Sigmoid.apply(x));

# Cost functions: cost(output, expected) ; costDerivative(layerInput, output, expected)

class L2Cost:
	@staticmethod
	def cost(output, expected): # On all the layer
		return np.sum((output - expected) ** 2) / 2

	@staticmethod
	def costDerivative(layerInput, output, expected): # Element by element
		return (output - expected) * Sigmoid.derivative(layerInput)

class CrossEntropyCost:
	@staticmethod
	def cost(output, expected): # On all the layer
		return -1 * np.sum(np.nan_to_num( expected * np.log(output) + (1 - expected) * np.log(1 - output) ))

	@staticmethod
	def costDerivative(layerInput, output, expected): # Element by element
		return output - expected

