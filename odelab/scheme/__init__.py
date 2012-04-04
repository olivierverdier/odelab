# -*- coding: UTF-8 -*-
from __future__ import division

"""
.. module :: scheme
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

The :class:`~odelab.scheme.Scheme` class contains methods on how to perform one iteration step.
It is its responsibility to take care of the time step.

Collection of schemes.
The main function of a :class:`odelab.scheme.Scheme` class is to define a :meth:`odelab.scheme.Scheme.step` which computes one step of the numerical solution.
"""

import numpy as np
import numpy.linalg


import odelab.newton as _rt


class Scheme(object):
	"""
General Scheme class. Subclass this class to define a specific integration method.
	"""
	def __init__(self, h=None, *args, **kwargs):
		if h is not None:
			self.h = h

	root_solver = _rt.FSolve

	def __repr__(self):
		try:
			h = self.h
			hs = "%.2e" % h
		except AttributeError:
			hs = "-"
		return '<%s: h=%s>' % (self.__class__.__name__, hs)

	tail_length = 1

	def delta(self, t,u0,h):
		"""
Compute the difference between current and next state.
		"""
		residual = self.get_residual(t,u0,h)
		guess = self.get_guess(t,u0,h)
		solver = _rt.MultipleSolver(residual)
		root = solver.run(guess)
		dt, du = self.reconstruct(root,t,u0,h)
		return dt, du

	def step(self, t,u0,h):
		"""
Implementation of the Compensated Summation algorithm as described in [HaLuWa06]_ §VIII.5.

.. [HaLuWa06] Hairer, Lubich, Wanner *Geometric Numerical Integration*, Springer, 2006.
		"""
		dt, du = self.delta(t,u0,h)
		self.roundoff += du
		u1 = u0 + self.roundoff
		self.roundoff += u0 - u1
		return t+dt, u1

	def do_step(self):
		event = self.last_event
		u,t = event[:-1], event[-1]
		new_t, new_u = self.step(t,u,self.h)
		self.last_event = np.hstack([new_u, new_t])
		return self.last_event

	def initialize(self, events):
		"""
Called the first time the scheme is used during a simulation.
		"""
		self.roundoff = 0.
		self.last_event = events[:,-1]


	def adjust_stepsize(self, error):
		"""
		Change the step size based on error estimation.
		To be overridden for a variable step size method.
		"""
		pass # todo: write a general implementation



	def get_residual(self, t,u0,h):
		"""
Return the residual, which root is u1 - u0.
		"""
		raise NotImplementedError()

	def get_guess(self,t,u0,h):
		"""
Default guess for the Newton iterations, assuming that the residual has the same size as u.
		"""
		return np.zeros_like(u0)

	def reconstruct(self, root,t,u0,h):
		"""
Default reconstruction function. It assumes that the root is already delta u.
		"""
		return h, root




class ode15s(Scheme):
	"""
	Simulation of matlab's ``ode15s`` solver.
	It is a BDF method of max order 5
	"""


	def __init__(self, **kwargs):
		super(ode15s,self).__init__(kwargs.get('h'))
		try:
			kwargs.pop('h')
		except KeyError:
			pass
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

