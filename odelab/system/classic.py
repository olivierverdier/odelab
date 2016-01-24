#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from odelab.system import System
import numpy as np

class VanderPol(System):
	r"""
The van der Pol oscillator, defined by:

.. math::
	u_0' &= u_1 \\
	u_1' &= μ (1-u_0^2)u_1 - u_0
	"""
	def labe(self, component):
		return ['x', 'v'][component]

	def __init__(self, mu=1.):
		self.mu = mu

	def f(self,t,y):
		return np.array([y[1], self.mu*(1-y[0]**2)*y[1] - y[0]])

