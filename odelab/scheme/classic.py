#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from . import Scheme
import numpy as np

class ExplicitEuler (Scheme):
	def delta(self, t, u0, h):
		return t + h, self.h*self.system.f(t, u0)

class ImplicitEuler (Scheme):
	def get_residual(self, t, u0, h):
		def residual(du):
			return du - h*self.system.f(t+h,u0+du)
		return residual

class RungeKutta4(Scheme):
	"""
	Runge-Kutta of order 4.
	"""
	def delta(self, t, u0, h):
		f = self.system.f
		Y1 = f(t, u0)
		Y2 = f(t + h/2., u0 + h*Y1/2.)
		Y3 = f(t + h/2., u0 + h*Y2/2.)
		Y4 = f(t + h, u0 + h*Y3)
		return t+h, h/6.*(Y1 + 2.*Y2 + 2.*Y3 + Y4)

class ExplicitTrapezoidal(Scheme):
	def delta(self,t,u0,h):
		f = self.system.f
		u1 = u0 + h*f(t,u0)
		res = h*.5*(f(t,u0) + f(t+h,u1))
		return t+h, res

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
		f = self.system.f
		Y1 = f(t, u0)
		Y2 = f(t + h/2., u0 + h*Y1/2.)
		Y3 = f(t + h/2, u0 + h*Y2/2)
		Z3 = f(t + h, u0 - h*Y1 + 2*h*Y2)
		Y4 = f(t + h, u0 + h*Y3)
		error = np.linalg.norm(h/6*(2*Y2 + Z3 - 2*Y3 - Y4))
		self.adjust_stepsize(error)
		return t+h, h/6*(Y1 + 2*Y2 + 2*Y3 + Y4)
