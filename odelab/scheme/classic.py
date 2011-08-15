#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from . import Scheme
import numpy as np

class ExplicitEuler (Scheme):
	def step(self, t, u, h):
		return t + h, u + self.h*self.system.f(t, u)

class ImplicitEuler (Scheme):
	def step(self, t, u, h):
		def residual(u1):
			return u1 - u - h*self.system.f(t+h,u1)
		N = self.root_solver(residual)
		u1 = N.run(u+h*self.system.f(t,u))
		return t + self.h, u1


class RungeKutta4 (Scheme):
	"""
	Runge-Kutta of order 4.
	"""
	def step(self, t, u, h):
		f = self.system.f
		Y1 = f(t, u)
		Y2 = f(t + h/2., u + h*Y1/2.)
		Y3 = f(t + h/2., u + h*Y2/2.)
		Y4 = f(t + h, u + h*Y3)
		return t+h, u + h/6.*(Y1 + 2.*Y2 + 2.*Y3 + Y4)

class ExplicitTrapezoidal(Scheme):
	def step(self,t,u,h):
		f = self.system.f
		u1 = u + h*f(t,u)
		res = u + h*.5*(f(t,u) + f(t+h,u1))
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

	def step(self, t, u, h):
		f = self.system.f
		Y1 = f(t, u)
		Y2 = f(t + h/2., u + h*Y1/2.)
		Y3 = f(t + h/2, u + h*Y2/2)
		Z3 = f(t + h, u - h*Y1 + 2*h*Y2)
		Y4 = f(t + h, u + h*Y3)
		error = np.linalg.norm(h/6*(2*Y2 + Z3 - 2*Y3 - Y4))
		self.adjust_stepsize(error)
		return t+h, u + h/6*(Y1 + 2*Y2 + 2*Y3 + Y4)
