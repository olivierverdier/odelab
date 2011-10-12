#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import odelab.scheme.rungekutta as rk

import numpy as np
from numpy import array

class System(object):
	"""
	General System class to define a simple dynamics given by a right-hand side.
	"""
	def __init__(self, f=None):
		if f is not None:
			self.f = f

	def label(self, component):
		return '%s' % component

	def preprocess(self, u0):
		return u0

	def postprocess(self, u1):
		return u1

class ODESystem(System):
	"""
	Simple wrapper to transform an ODE into a semi-explicit DAE.
	"""
	def __init__(self, f, RK_class=rk.LobattoIIIA):
		self.f = f
		self.RK_class = RK_class

	def state(self, u):
		return u[:1]

	def lag(self,u):
		return u[1:] # should be empty

	def multi_dynamics(self, tu):
		return {self.RK_class: array([self.f(tu)])}

	def constraint(self, tu):
		return np.zeros([0,np.shape(tu)[1]])


