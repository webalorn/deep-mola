class PresetGroup():
	def __init__(self):
		self.layers = []
		self.inLayers = [] # Layers that must be connected to previous layers
		self.outLayers = [] # Layers that will pass informations to next layers
		self.endLayers = [] # Layers that represent an output of the network
		self.loss = None

	def withoutEnds(self): # To remove end layers if we don't want to use them
		self.endLayers = []
		return self

	def withLoss(self, loss):
		self.loss = loss

	def add(self, layer, isIn=False, isOut=False, isEnd=False):
		if isinstance(layer, list):
			for l in layer:
				self.add(l, isIn, isOut, isEnd)
		else:
			self.layers.append(layer)
			if isIn:
				self.inLayers.append(layer)
			if isOut:
				self.outLayers.append(layer)
			if isEnd:
				self.endLayers.append(layer)

	"""
		Functions to use the group as a layer when defining the network
	"""
	def withInput(self, layer):
		for l in self.inLayers:
			l.addInput(layer)
		return self

	def asOutput(self, loss=None):
		for l in self.endLayers:
			self.network.addOutput(l, loss)
		return self