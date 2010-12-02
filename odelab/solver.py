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

import scipy.io # used for saving in shelves
import numpy as np
import pylab as PL

import itertools


class Solver (object):
	"""
	General Solver class, that takes care of calling the step function and storing the intermediate results.

	:Parameters:
		system : :class:`odelab.system.System`
			Object describing the system. The requirement on that class may vary. See the documentation of the various solver subclasses. The system may also be specified later, although before any simulation of course.
	"""

	def __init__(self, system=None):
		self.system = system


	# default values for the total time
	time = 1.

	def initialize(self, u0=None, t0=0, h=None, time=None):
		"""
Initialize the solver to the initial condition :math:`u(t0) = u0`.

:param array u0: initial condition; if it is not provided, it is set to the previous initial condition.
:param scalar t0: initial time
:param scalar h: time step
:param scalar time: span of the simulation
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

	auto_save = False # whether to automatically save the session after a run; especially useful for tests

	def __exit__(self, ex_type, ex_value, traceback):
		self.ats = np.array(self.ts)
		self.aus = np.array(self.us).T
		if self.auto_save:
			self.save()

	catch_runtime = True # whether to catch runtime exception (not catching allows to see the traceback)

	t_tol = 1e-12 # tolerance to tell whether the final time is reached

	def run(self, time=None):
		"""
		Run the simulation for a given time.

:param scalar time: the time span for which to run; if none is given, the default ``self.time`` is used
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
					if self.catch_runtime:
						raise self.Runtime('%s raised after %d steps: %s' % (type(e).__name__,i,e.args), e, i)
					else:
						raise
				else:
					if np.any(np.isnan(u)):
						raise self.Unstable('Unstable after %d steps.' % i)

					self.ts.append(t)
					self.us.append(u)
					if t > tf - self.t_tol:
						break
			else:
				raise self.FinalTimeNotReached("Reached maximal number of iterations: {0}".format(self.max_iter))

	def guess_name(self):
		"""
		Guess a name for this session.
		"""
		guess = "{system}_{scheme}_T{time}_N{nsteps}".format(system=type(self.system).__name__, scheme=type(self.scheme).__name__, time=self.time, nsteps=len(self.us))
		sanitized = guess.replace('.','_')
		return sanitized

	shelf_name = 'bank' # default shelf name

	def save(self, name=None):
		"""
		Save the current results in a scipy shelf.
		"""
		shelf_name = name or self.guess_name()
		print shelf_name
		scipy.io.save_as_module(self.shelf_name, {shelf_name: self})

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

	max_plot_res = 500 # max plot resolution

	def plot(self, components=None, plot_exact=True, error=False, save=None, **plot_args):
		"""
		Plot some components of the solution.

:param list components: either a given component of the solution, or a list of components to plot, or a list of strings corresponding to methods of the system.
:param boolean plot_exact: whether to plot the exact solution (if available)
:param string save: whether to save the plot in a file
:param boolean error: to plot (the log10 of) the error instead of the value itself
		"""
		# some sampling
		size = len(self.ts)
		stride = np.ceil(size/self.max_plot_res)
		ats = self.ats[::stride]
		aus = self.aus[:,::stride]

		# components
		if components is None: # plot all the components by default
			components = range(len(self.us[0]))
		if not np.iterable(components): # makes the code work when plotting a single component
			components = [components]

		if plot_exact or error:
			has_exact = hasattr(self.system, 'exact')
			if has_exact:
				exact = self.system.exact(ats, self.initial())
		compute_exact = (plot_exact or error) and has_exact
		axis = PL.gca()
		if save: # if this is meant to be saved, clear the previous plot first
			axis.cla()
		previous_line = len(axis.lines)

		ut = np.vstack([aus, ats])
		for component in components:
			if isinstance(component, str):
				label = component
				data = self.system.__getattribute__(component)(ut)
				if compute_exact:
					exact_comp = self.system.__getattribute__(component)(np.vstack([exact, ats]))
			else:
				label = self.system.label(component)
				data = aus[component]
				if compute_exact:
					exact_comp = exact[component]

			# set up plot arguments
			defaults = {'ls':'-', 'marker':','}
			defaults.update(plot_args)

			if error and has_exact:
				data = np.log10(np.abs(data - exact_comp))
			axis.plot(ats, data, ',-', label=label, **defaults)
			if compute_exact and not error:
				axis._get_lines.count -= 1
				axis.plot(ats, exact_comp, ls='-', lw=2, label='%s*' % label)
		axis.set_xlabel('time')
		axis.legend()
		if save:
			PL.savefig(save, format='pdf', **plot_args)
		else:
			PL.plot() # plot only in interactive mode
		return axis

	def plot_function(self, function, *args, **kwargs):
		"""
		Plot a given function of the state. May be useful to plot constraints or energy.

		This is now a convenience function that calls `odelab.solver.plot`.

:param string function: name of the method to call on the current system object

		:Example:
			the code::

				solver.plot_function('energy')

			will call the method ``solver.system.energy`` on the current stored solution points.
		"""
		return self.plot(*args, components=[function], **kwargs)

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
	def __init__(self, scheme, system=None):
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







