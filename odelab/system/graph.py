#!/usr/bin/env python
# coding: utf-8
from __future__ import division


import odelab.scheme.rungekutta as rk
from odelab.system import System
import numpy as np
from numpy import array


class GraphSystem(System):
	r"""
	Trivial semi-explicit index 2 DAE of the form:

.. math::

		x' = 1\\
		y' = λ\\
		y  = f(x)
	"""

	def label(self, component):
		return ['x','y',u'λ'][component]

	def state(self, u):
		return u[:2]

	def lag(self, u):
		return u[2:3]

	def multi_dynamics(self, ut):
		x,y = self.state(ut)
		return {
			rk.RadauIIA: array([np.ones_like(x), self.lag(ut)[0]]),
			}

	def dynamics(self, ut):
		x,y = self.state(ut)
		return array([np.ones_like(x), self.lag(ut)[0]]),

	def constraint(self, ut):
		x,y = self.state(ut)
		t = ut[-1]
		return array([y - self(t, x)])

	def hidden_error(self, t, u):
		return self.lag(u)[0] - self.f.der(t,self.state(u)[0])

	def exact(self, t, u0):
		x = u0[0] + t
		return array([x, self.f(x), self.f.der(x)])

class ExpGraphSystem(GraphSystem):
	def dynamics(self,ut):
		x,y = self.state(ut)
		return array([-x, self.lag(ut)[0]])

	def exact(self, t, u0):
		x = u0[0]*np.exp(-t)
		return array([x, self.f(x), self.f.der(x)])

class QuasiGraphSystem(GraphSystem):
	u"""
	The quasi-graph system is obtained from the graph system by the variable change:

.. math::

	z = x + xy

	The equations thus become:

.. math::

	z' = 1 + y + \frac{z}{1+y)λ
	y' = λ
	y = f(x)

	"""
	def label(self, component):
		return ['z','y',u'λ'][component]

	def constraint(self,ut):
		z,y = self.state(ut)
		x = z/(1+y)
		xystate = array([x,y,ut[-1]])
		return super(QuasiGraphSystem,self).constraint(xystate)

	def dynamics(self, ut):
		z,y = self.state(ut)
		return array([1 + y + z/(1+y)*self.lag(ut)[0], self.lag(ut)[0]]),

	def exact(self,t,u0):
		x0 = u0[0]
		x = x0 + t
		y = self(t, x)
		z = x*(1+y)
		return array([z, y, self._f.der(t, x)])


