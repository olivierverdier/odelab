#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy as np

from . import NonHolonomic

class Pendulum(NonHolonomic):
	def __init__(self, constraint=None, codistribution=None):
		self._constraint = constraint
		self._codistribution = codistribution

	def position(self, u):
		return u[:2]

	def velocity(self, u):
		return u[2:4]

	def lag(self,u):
		return u[4:5]

	def codistribution(self, u):
		return self._codistribution(u)

	def force(self, u):
		return -np.array([0,1])

	def average_force(self,u0,u1):
		return self.force(u0)

	def constraint(self, u):
		return self._constraint(u)

class CirclePendulum(Pendulum):
	def codistribution(self, u):
		return self.position(u).reshape(1,-1)

	def constraint(self,u):
		x,y =  u[0], u[1]
		return np.sqrt(x*x + y*y)

	def energy(self, u):
		q = self.position(u)
		v = self.velocity(u)
		v2 = np.square(v)
		return (v2[0] + v2[1])/2 + q[1]

class SinePendulum(Pendulum):
	def codistribution(self, u):
		x,y = self.position(u)
		return np.hstack([np.cos(x), -np.ones_like(y)]).reshape(1,-1)

	def constraint(self, u):
		x,y = self.position(u)
		return y - np.sin(x)

class PPendulum(Pendulum):
	def __init__(self, order):
		self.order = order

	def constraint(self,u):
		x,y = self.position(u)
		p = self.order
		return x**p + y**p

	def codistribution(self, q):
		return q**(self.order -1).reshape(1,-1)
