from dml.nnet.models.network import Network

class Sequential(Network):
	"""
		Automaticly add previous layer as input for layers without input
		If not output is specified for the network, use the last layer as output
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def build(self, *args, **kwargs):
		if not self.outputLayers:
			self.outputLayers = [self.layers[-1]]

		for l1, l2 in zip(self.layers, self.layers[1:]):
			if not l2.inputs and l2.nbInputs:
				l2.inputs = [l1]

		super().build(*args, **kwargs)