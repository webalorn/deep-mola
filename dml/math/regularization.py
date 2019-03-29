class Regulator:
	pass

class L2regul(Regulator):
	def __init__(self, regulRate):
		self.regulRate = regulRate

	def cost(self, params):
		return sum([(p ** 2).sum() for p in params]) * self.regulRate * 0.5