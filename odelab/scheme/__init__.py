# -*- coding: UTF-8 -*-
from __future__ import division

"""
Collection of schemes. The main function of a :class:`odelab.scheme.Scheme` class is to define a :meth:`odelab.scheme.Scheme.step` which computes one step of the numerical solution.
"""

import numpy as np
import numpy.linalg

import logging

import odelab.newton as _rt

class Scheme(object):

	root_solver = _rt.FSolve

	def __repr__(self):
		try:
			h = self.h
			hs = "%.2e" % h
		except AttributeError:
			hs = "-"
		return '<%s: h=%s>' % (self.__class__.__name__, hs)

	tail_length = 1

	@property
	def system(self):
		return self.solver.system

	def __getstate__(self):
		"""
		Make sure that the solver attribute is never pickled.
		"""
		d = self.__dict__.copy()
		try:
			del d['solver']
		except KeyError:
			pass
		return d

	def increment_stepsize(self):
		"""
		Change the step size based on error estimation.
		To be overridden for a variable step size method.
		"""
		pass

	h_default = .01

	# to be removed; use a signal instead
	def get_h(self):
		return self._h
	def set_h(self, h):
		self._h = h
		self._h_dirty = True
	h = property(get_h, set_h)

	def initialize(self, events):
		try:
			self.h = self.solver.h
		except AttributeError:
			self.h = self.h_default
		self.roundoff = 0.

	def delta(self, t,u0,h):
		"""
Compute the difference between current and next state.
		"""
		residual = self.delta_residual(t,u0,h)
		fsolve = _rt.FSolve(residual)
		newton = _rt.Newton(residual)
		try:
			du = fsolve.run(np.zeros_like(u0))
		except fsolve.DidNotConverge:
			logging.info("Switch nonlinear solver")
			du = newton.run(np.zeros_like(u0))
		return t+h, du

	def step(self, t,u0,h):
		"""
Implementation of the Compensated Summation algorithm as described in _[HaLuWa2006].
		"""
		t1, du = self.delta(t,u0,h)
		self.roundoff += du
		u1 = u0 + self.roundoff
		self.roundoff += u0 - u1
		return t1, u1


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

	def increment_stepsize(self):
		if self.error > 1e-15:
			self.h *= (self.tol/self.error)**(1/self.error_order)
		else:
			self.h = 1.

	def step(self, t, u, h):
		f = self.system.f
		Y1 = f(t, u)
		Y2 = f(t + h/2., u + h*Y1/2.)
		Y3 = f(t + h/2, u + h*Y2/2)
		Z3 = f(t + h, u - h*Y1 + 2*h*Y2)
		Y4 = f(t + h, u + h*Y3)
		self.error = np.linalg.norm(h/6*(2*Y2 + Z3 - 2*Y3 - Y4))
		return t+h, u + h/6*(Y1 + 2*Y2 + 2*Y3 + Y4)

class ode15s(Scheme):
	"""
	Simulation of matlab's ``ode15s`` solver.
	It is a BDF method of max order 5
	"""


	def __init__(self, **kwargs):
		self.integrator_kwargs = kwargs

	def initialize(self, events): # the system must be defined before this is called!
		super(ode15s,self).initialize(events)
		import scipy.integrate
		self.integ = scipy.integrate.ode(self.system.f)
		e0 = events[:,-1] # duplicate code with Solver.final
		vodevariant = ['vode', 'zvode'][np.iscomplexobj(e0)]
		self.integ.set_integrator(vodevariant, method='bdf', order=5, nsteps=3000, **self.integrator_kwargs)
		self.integ.set_initial_value(e0[:-1], e0[-1])

	def step(self, t, u, h):
		self.integ.integrate(self.integ.t + h)
		if not self.integ.successful():
			print("vode error")
		return self.integ.t, self.integ.y

