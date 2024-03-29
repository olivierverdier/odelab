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

