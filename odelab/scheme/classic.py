#!/usr/bin/env python
# coding: utf-8
from __future__ import division


from . import Scheme
import numpy as np

class ExplicitEuler (Scheme):
	"""
Explicit version of the Euler method, defined by:

.. math::
	u_1 = u_0 + hf(t_0, u_0)
	"""
	def delta(self, t, u0, h):
		return h, self.h*self.system(t, u0)

class ImplicitEuler (Scheme):
	"""
The standard implicit Euler scheme, defined by:

.. math::
	u_1 = u_0 + hf(t_1, u_1)
	"""
	def get_residual(self, t, u0, h):
		def residual(du):
			return du - h*self.system(t+h,u0+du)
		return residual

class ImplicitMidPoint(Scheme):
	"""
The implicit mid point rule

.. math::
	u_1 = u_0 + hf((t_0 + t_1)/2, (u_0 + u_1)/2)
	"""
	def get_residual(self, t, u0, h):
		def residual(du):
			mid = u0 + du/2
			return du - h*self.system(t+h/2, mid)
		return residual

class RungeKutta4(Scheme):
	"""
Standard Runge-Kutta of order 4.
	"""
	def delta(self, t, u0, h):
		f = self.system
		Y1 = f(t, u0)
		Y2 = f(t + h/2., u0 + h*Y1/2.)
		Y3 = f(t + h/2., u0 + h*Y2/2.)
		Y4 = f(t + h, u0 + h*Y3)
		return h, h/6.*(Y1 + 2.*Y2 + 2.*Y3 + Y4)

class ExplicitTrapezoidal(Scheme):
	def delta(self,t,u0,h):
		f = self.system
		u1 = u0 + h*f(t,u0)
		res = h*.5*(f(t,u0) + f(t+h,u1))
		return h, res

class RungeKutta34 (Scheme):
	"""
Adaptive Runge-Kutta of order four.
	"""
	error_order = 4.
	# default tolerance
	tol = 1e-6

	def adjust_stepsize(self, error):
		if error > 1e-15:
			self.h *= (self.tol/error)**(1/self.error_order)
		else:
			self.h = 1.

	def delta(self, t, u0, h):
		f = self.system
		Y1 = f(t, u0)
		Y2 = f(t + h/2., u0 + h*Y1/2.)
		Y3 = f(t + h/2, u0 + h*Y2/2)
		Z3 = f(t + h, u0 - h*Y1 + 2*h*Y2)
		Y4 = f(t + h, u0 + h*Y3)
		error = np.linalg.norm(h/6*(2*Y2 + Z3 - 2*Y3 - Y4))
		self.adjust_stepsize(error)
		return h, h/6*(Y1 + 2*Y2 + 2*Y3 + Y4)
