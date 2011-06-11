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

import itertools

from odelab.plotter import Plotter

import tables

class Solver (object):
	"""
	General Solver class, that takes care of calling the step function and storing the intermediate results.

	:Parameters:
		system : :class:`odelab.system.System`
			Object describing the system. The requirement on that class may vary. See the documentation of the various solver subclasses. The system may also be specified later, although before any simulation of course.
	"""

	def __init__(self, system=None, file=None, name=None):
		self.system = system
		if file is None:
			import tempfile
			f = tempfile.NamedTemporaryFile(delete=True)
			self.file = tables.openFile(f.name, mode='a')
			# the following is to avoid PyTables to keep a reference on the open file
			# http://thread.gmane.org/gmane.comp.python.pytables.user/1100/focus=1107
			# it is unsatisfactory: perhaps the best way is to use __enter__ and __exit__ appropriately
			del tables.file._open_files[f.name]
		else:
			self.file = file
		self.name = name or 'events'


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

		# first remove the events node if they exist
		try:
			self.load_data()
		except tables.NoSuchNodeError:
			pass
		else:
			self.events.remove()

		compression = tables.Filters(complevel=1, complib='blosc', fletcher32=True)
		self.events = self.file.createEArray(
				where=self.file.root,
				name=self.name,
				atom=tables.Atom.from_dtype(event0.dtype),
				shape=(len(event0),0),
				filters=compression)
		self.events.append(np.array([event0]).reshape(-1,1)) # todo: factorize the call to reshape, append

		if h is not None:
			self.h = h
		if time is not None:
			self.time = time

	def __len__(self):
		return self.events.nrows

	def load_data(self):
		"""
		"""
		self.events = self.file.getNode('/'+self.name)

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


	max_iter = 1000000
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
		event = self.events[:,-1]
		generator = self.generate(event)
		return generator

	auto_save = False # whether to automatically save the session after a run; especially useful for tests

	def __exit__(self, ex_type, ex_value, traceback):
		self.file.flush()
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
			t0 = self.events[-1,-1]
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

					self.events.append(event.reshape(-1,1))
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
		event = self.events[:,index]
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

	def plot(self, components=None, plot_exact=True, error=False, save=None, time_component=None, **plot_args):
		plotter = Plotter(self)
		return plotter.plot(components, plot_exact, error, save, time_component, **plot_args)

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
	def __init__(self, scheme, *args, **kwargs):
		super(SingleStepSolver, self).__init__(*args, **kwargs)
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

	def __init__(self, scheme, single_step_scheme=None, *args, **kwargs):
		super(MultiStepSolver, self).__init__(scheme, *args, **kwargs)
		if single_step_scheme is None:
			from odelab.scheme.exponential import HochOst4
			single_step_scheme = HochOst4() # hard coded for the moment
		self.single_step_scheme = single_step_scheme
		self.single_step_scheme.solver = self







