#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.system.base import *

class VanderPol(System):
	def __init__(self, mu=1.):
		self.mu = mu

	def f(self,t,y):
		return np.array([y[1], self.mu*(1-y[0]**2)*y[1] - y[0]])

