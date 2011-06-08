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
		if u0 is not None: # initial condition provided
			if np.isscalar(u0):
				u0 = [u0]
			u0 = np.array(u0)
			raw_event0 = np.hstack([u0,t0])
			event0 = self.system.preprocess(raw_event0)
		else: # start from the previous initial conditions
			try:
				event0 = self.events[0]
			except AttributeError:
				raise self.NotInitialized("You must provide an initial condition.")
		self.events = [event0]
		if h is not None:
			self.h = h
		if time is not None:
			self.time = time

	def __len__(self):
		return len(self.events)

	def load_data(self, data):
		"""
Initialize the solver from previously saved data.
:param array data: event array, with the same format as :py:attr:`events_array`
		"""
		self.events = list(data.T)
		self.update_events_array()

	def update_events_array(self):
		self.events_array = np.array(self.events).T

	def generate(self, event):
		"""
		Generates the (t,u) values.
		"""
		u,t = event[:-1], event[-1]
		for i in itertools.count(): # infinite loop
			t, u = self.step(t, u)
			event = np.hstack([u,t])
			yield event
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
		event = self.events[-1]
		generator = self.generate(event)
		return generator

	auto_save = False # whether to automatically save the session after a run; especially useful for tests

	def __exit__(self, ex_type, ex_value, traceback):
		self.update_events_array()
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
			t0 = self.events[-1][-1]
		except AttributeError:
			raise self.NotInitialized("You must call the `initialize` method before you can run the solver.")
		tf = t0 + time # final time
		with self as generator:
			for i in xrange(self.max_iter):
				try:
					event = next(generator)
				except Exception as e:
					if self.catch_runtime:
						raise self.Runtime('%s raised after %d steps: %s' % (type(e).__name__,i,e.args), e, i)
					else:
						raise
				else:
					if np.any(np.isnan(event)):
						raise self.Unstable('Unstable after %d steps.' % i)

					self.events.append(event)
					if event[-1] > tf - self.t_tol:
						break
			else:
				raise self.FinalTimeNotReached("Reached maximal number of iterations: {0}".format(self.max_iter))

	def guess_name(self):
		"""
		Guess a name for this session.
		"""
		guess = "{system}_{scheme}_T{time}_N{nsteps}".format(system=type(self.system).__name__, scheme=type(self.scheme).__name__, time=self.time, nsteps=len(self.events))
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

	def get_u(self, index, process=True):
		"""
		Return u[index] after post-processing.
		"""
		event = self.events[index]
		if process:
			event = self.system.postprocess(event)
		return event

	def get_times(self):
		return self.events[-1]

	def final_time(self):
		return self.get_times()[-1]

	def initial(self, process=True):
		"""
		Convenience method to obtain the initial condition.
		"""
		return self.get_u(0, process)

	def final(self, process=True):
		"""
		Convenience method to obtain the last computed value.
		"""
		return self.get_u(-1, process)

	max_plot_res = 500 # max plot resolution

	def plot(self, components=None, plot_exact=True, error=False, save=None, time_component=None, **plot_args):
		"""
		Plot some components of the solution.

:param list components: either a given component of the solution, or a list of components to plot, or a list of strings corresponding to methods of the system.
:param boolean plot_exact: whether to plot the exact solution (if available)
:param string save: whether to save the plot in a file
:param boolean error: to plot (the log10 of) the error instead of the value itself
:param int time_component: component to use as x variable (if None, then time is used)
		"""
		# set up plot arguments
		defaults = {'ls':'-', 'marker':','}
		defaults.update(plot_args)

		axis = PL.gca()
		if save: # if this is meant to be saved, clear the previous plot first
			axis.cla()
		for pts in self.generate_plot_data(components, plot_exact, error, save, time_component):
			data = pts['data']
			ats = pts['x']
			label = pts['label']
			axis.plot(ats, data, ',-', label=label,  **defaults)
			exact = pts.get('exact', None)
			if exact is not None:
				current_color = axis.lines[-1].get_color() # figure out current colour
				axis.plot(ats, exact, ls='-', lw=2, label='{}*'.format(label), color=current_color)

		axis.set_xlabel(pts['time_label'])
		axis.legend()
		if save:
			PL.savefig(save, format='pdf', **plot_args)
		else:
			PL.plot() # plot only in interactive mode
		return axis

	def generate_plot_data(self, components=None, plot_exact=True, error=False, save=None, time_component=None):
		# some sampling
		size = len(self.events)
		stride = np.ceil(size/self.max_plot_res)
		events = self.events_array[:,::stride]
		ats = events[-1]

		# components
		if components is None: # plot all the components by default
			components = range(self.events[0].size - 1)
		if not np.iterable(components): # makes the code work when plotting a single component
			components = [components]
		if time_component is not None:
			components.insert(0,time_component)

		if plot_exact or error:
			sys_exact = getattr(self.system, 'exact', None)
			if sys_exact:
				exact = sys_exact(ats, self.initial())
		compute_exact = (plot_exact or error) and sys_exact

		time_label = 'time'

		for component_i, component in enumerate(components):
			if isinstance(component, str):
				label = component
				function = getattr(self.system, component)
				data = function(events)
				if compute_exact:
					exact_comp = function(np.vstack([exact, ats]))
			else:
				label = self.system.label(component)
				data = events[component]
				if compute_exact:
					exact_comp = exact[component]

			if error and sys_exact:
				data = np.log10(np.abs(data - exact_comp))
			if time_component is not None and not component_i:
				# at the first step, if time_component is not time, then replace the time vector by the desired component
				ats = data
				time_label = label
				continue
			else:
				pts = {}
				pts['x'] = ats
				pts['time_label'] = time_label
				pts['data'] = data
				pts['label'] = label
				if compute_exact and not error:
					pts['exact'] = exact_comp
				yield pts

	def plot_function(self, function, *args, **kwargs):
		"""
		Plot a given function of the state. May be useful to plot constraints or energy.

		This is now a convenience function that calls `odelab.solver.plot`.

:param string function: name of the method to call on the current system object

		:Example:
			the code ``solver.plot_function('energy')`` will call the method ``solver.system.energy`` on the current stored solution points.
		"""
		return self.plot(*args, components=[function], **kwargs)

	def plot2D(self, time_component=0, other_component=1, *args, **kwargs):
		"""
		Plot components vs another one
		"""
		self.plot(other_component, time_component=time_component, *args, **kwargs)

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
		stage = len(self)
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







