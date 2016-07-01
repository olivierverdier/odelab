#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.system import System
import numpy as np

class Exponential(System):
	def __init__(self, nonlin, L):
		self.L = L
		self.nonlin = nonlin

	def linear(self):
		return self.L

	def __call__(self, t, u):
		return np.dot(self.linear(), u) + self.nonlin(t,u)

def zero_dynamics(t,u):
	return np.zeros_like(u)

class Linear(Exponential):
	def __init__(self, L):
		super(Linear, self).__init__(zero_dynamics, L)

class NoLinear(Exponential):
	def __init__(self, f, size):
		super(NoLinear,self).__init__(f, np.zeros([size,size]))


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

