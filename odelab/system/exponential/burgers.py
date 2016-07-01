import numpy as np

from . import Exponential

class Burgers(Exponential):
	def __init__(self, viscosity=0.03, size=128):
		self.viscosity = viscosity
		self.size = size
		self.initialize()

	def linear(self):
		return -self.laplace



class BurgersComplex(Burgers):
	def __getstate__(self):
		state = self.__dict__.copy()
		del state['laplace']
		del state['points']
		del state['k']
		return state

	def initialize(self):
		self.k = 2*np.pi*np.array(np.fft.fftfreq(self.size, 1/self.size),dtype='complex')
		self.k[len(self.k)/2] = 0
		self.laplace = self.viscosity*np.diag(self.k**2)
		x = np.linspace(0,1,self.size,endpoint=False)
		self.points = x - x.mean() # too lazy to compute x.mean manually here...

	def preprocess(self, event0):
		return np.hstack([np.fft.fft(event0[:-1]), event0[-1]])

	def postprocess(self, event1):
		return np.hstack([np.real(np.fft.ifft(event1[:-1])),event1[-1]])

	def nonlin(self, t, u):
		return -0.5j * self.k * np.fft.fft(np.real(np.fft.ifft(u)) ** 2)
