# -*- coding: UTF-8 -*-
"""
:mod:`Solver` -- ODE Solvers
============================

A collection of solvers for ODEs of various types.

.. module :: solver
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

The :class:`Scheme` class contains methods on how to perform one iteration step. 
It is its responsibility to take care of the time step.

The higher level class is :class:`Solver`, which is initialized with an instance of a :class:`Scheme` class.
"""
from __future__ import division

import numpy as np
import pylab as PL

from odelab.newton import Newton, FSolve

import itertools


class Solver (object):
	"""
	General Solver class, that takes care of calling the step function and storing the intermediate results.
	
	:Parameters:
		system : :class:`System`
			Object describing the system. The requirement on that class may vary. See the documentation of the various solver subclasses. The system may also be specified later, although before any simulation of course.
	"""

	def __init__(self, system=None):
		self.system = system
	
	
	# default values for the total time
	time = 1.

	def initialize(self, u0=None, t0=0, h=None, time=None):
		"""
Initialize the solver to the initial condition :math:`u(t0) = u0`.

:type u0: array
:param u0: initial condition; if it is not provided, it is set to the previous initial condition.
:type t0: scalar
:param t0: initial time
:type h: scalar
:param h: time step
:type time: scalar
:param time: span of the simulation
		"""
		if u0 is not None:
			if np.isscalar(u0):
				u0 = [u0]
			u0 = np.array(u0)
			u0 = self.system.preprocess(u0)
		else: # start from the previous initial conditions
			try:
				u0 = self.us[0]
				t0 = self.ts[0]
			except AttributeError:
				raise self.NotInitialized("You must provide an initial condition.")
		self.ts = [t0]
		self.us = [u0]
		if h is not None:
			self.h = h
		if time is not None:
			self.time = time


	def generate(self, t, u):
		"""
		Generates the (t,u) values.
		"""
		for i in itertools.count(): # infinite loop
			t, u = self.step(t, u)
			yield t, u
			self.increment_stepsize()
	

	max_iter = 100000
	class FinalTimeNotReached(Exception):
		"""
		Raised when the final time was not reached within the given ``max_iter`` number of iterations.
		"""
	
	class Unstable(Exception):
		"""
		Raised when the scheme produces NaN values.
		"""
	
	class NotInitialized(Exception):
		"""
		Raised when the solver is not properly initialized.
		"""
	
	class Runtime(Exception):
		"""
		Raised to relay an exception occurred while running the solver.
		"""
	
	def simulating(self):
		return self
	
	def __enter__(self):
		# start from the last time we stopped
		t = t0 = self.ts[-1]
		u = self.us[-1]
		generator = self.generate(t, u)
		return generator
		
	def __exit__(self, ex_type, ex_value, traceback):
		self.ats = np.array(self.ts)
		self.aus = np.array(self.us).T

	t_tol = 1e-12 # tolerance to tell whether the final time is reached
	
	def run(self, time=None):
		"""
		Run the simulation for a given time.
		
		:Parameters:
			time : scalar
				the time span for which to run; if none is given, the default ``self.time`` is used
		"""
		if time is None:
			time = self.time
		try:
			t0 = self.ts[-1]
		except AttributeError:
			raise self.NotInitialized("You must call the `initialize` method before you can run the solver.")
		tf = t0 + time # final time
		with self as generator:
			for i in xrange(self.max_iter):
				try:
					t,u = next(generator)
				except Exception as e:
					raise self.Runtime('%s raised after %d steps: %s' % (type(e).__name__,i,e.args), e, i)
				if np.any(np.isnan(u)):
					raise self.Unstable('Unstable after %d steps.' % i)

				self.ts.append(t)
				self.us.append(u)
				if t > tf - self.t_tol:
					break
			else:
				raise self.FinalTimeNotReached("Reached maximal number of iterations: {0}".format(self.max_iter))

	def get_u(self, index):
		"""
		Return u[index] after post-processing.
		"""
		return self.system.postprocess(self.us[index])

	def initial(self):
		"""
		Convenience method to obtain the initial condition.
		"""
		return self.get_u(0)
	
	def final(self):
		"""
		Convenience method to obtain the last computed value.
		"""
		return self.get_u(-1)
	
	def plot(self, components=None, **plot_args):
		"""
		Plot some components of the solution.
		
		:Parameters:
			components : scalar|array_like
				either a given component of the solution, or a list of components to plot.
		"""
		if components is None:
			components = range(len(self.us[0]))
		if not np.iterable(components):
			components = [components]
		has_exact = hasattr(self.system, 'exact')
		if has_exact:
			exact = self.system.exact(self.ats)
		axis = PL.gca()
		previous_line = len(axis.lines)
		for component in components:
			label = self.system.label(component)
			defaults = {'ls':'-', 'marker':','}
			defaults.update(plot_args)
			axis.plot(self.ats, self.aus[component], ',-', label=label, **defaults)
			if has_exact:
				axis._get_lines.count -= 1
				axis.plot(self.ats, exact[component], ls='-', lw=2, label='%s_' % label)
		axis.set_xlabel('time')
		axis.legend()
		return axis.lines[previous_line:]
	
	def plot_function(self, function, **plot_args):
		"""
		Plot a given function of the state. May be useful to plot constraints or energy.
		
		:param function: name of the method to call on the current system object
		:type function: string
		
		:Example:
			the code::
			
				solver.plot_function('energy')
			
			will call the method ``solver.system.energy`` on the current stored solution points.
		"""
		values = self.system.__getattribute__(function)(np.vstack([self.aus, self.ats]))
		return PL.plot(self.ats, values.T, label=function, **plot_args)

	def plot2D(self):
		"""
		Plot ux vs uy
		"""
		PL.plot(self.aus[:,0],self.aus[:,1], '.-')
		PL.xlabel('ux')
		PL.ylabel('uy')
	
	quiver_res = 20
	def quiver(self):
		mins = self.aus.min(axis=0)
		maxs = self.aus.max(axis=0)
		X,Y = np.meshgrid(linspace(mins[0], maxs[0], self.quiver_res), 
								linspace(mins[1], maxs[1], self.quiver_res))
		Z = np.dstack([X,Y])
		vals = self.f(0,Z.transpose(2,0,1))
		PL.quiver(X,Y,vals[0], vals[1])

class SingleStepSolver(Solver):
	def __init__(self, scheme, system):
		super(SingleStepSolver, self).__init__(system)
		self.scheme = scheme
	
	def __repr__(self):
		return '<%s: %s>' % ('Solver', str(self.scheme))
	
	def set_scheme(self, scheme):
		self.current_scheme = scheme
		self.current_scheme.solver = self
		self.current_scheme.initialize()
	
	def step_current(self, t,u):
		return self.current_scheme.step(t,u)
	
	def step(self, t,u):
		stage = len(self.us)
		if stage < self.scheme.tail_length: # not enough past values to run main scheme
			if stage == 1:
				self.set_scheme(self.single_step_scheme)
		if stage == self.scheme.tail_length: # main scheme kicks in
			self.set_scheme(self.scheme)
		return self.step_current(t,u)
	
	def increment_stepsize(self):
		self.current_scheme.increment_stepsize()

class MultiStepSolver(SingleStepSolver):
## 	default_single_step_scheme = HochOst4()
	
	def __init__(self, scheme, system, single_step_scheme=None):
		super(MultiStepSolver, self).__init__(scheme, system)
		if single_step_scheme is None:
			from odelab.scheme.exponential import HochOst4
			single_step_scheme = HochOst4() # hard coded for the moment
		self.single_step_scheme = single_step_scheme
		self.single_step_scheme.solver = self
	



	


