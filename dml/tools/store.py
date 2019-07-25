from importlib import import_module
import os, json

# Serializable objects

class Serializable:
	def serialize(self):
		return {
			'class': self.__class__.__qualname__,
			'module': self.__class__.__module__,
		}

	@classmethod
	def recreate(cls, datas):
		obj = cls.reacretObj(datas)
		obj.repopulate(datas)
		return obj

	@classmethod
	def reacretObj(cls, datas):
		return cls()

	def repopulate(self, datas):
		pass


class Storable(Serializable):
	def saveTo(self, file):
		""" file can be a string or a file object """
		if isinstance(file, str):
			os.makedirs(os.path.dirname(file), exist_ok=True)
			file = open(file, 'w')

		json.dump(self.serialize(), file)

	@classmethod
	def loadFrom(cls, file):
		return loadModelFrom(file)

# Functions to serialize an object

def serializeFunc(func):
	return {
		'function': func.__qualname__,
		'module': func.__module__,
	}

# Functions to recreate an object from serialized datas

def importObject(module, name):
	return import_module(module).__getattribute__(name)

def recreateClass(datas):
	newClass = importObject(datas['module'], datas['class'])
	return newClass.recreate(datas)

def recreateFunc(datas):
	return importObject(datas['module'], datas['function'])

def recreateObject(datas):
	if datas == None:
		return None
	elif 'class' in datas:
		return recreateClass(datas)
	else:
		return recreateFunc(datas)

def loadModelFrom(file):
	if isinstance(file, str):
		file = open(file, 'r')

	return recreateObject(json.load(file))