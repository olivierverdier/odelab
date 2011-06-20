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
import warnings
import time

from odelab.plotter import Plotter

import tables

from contextlib import contextmanager

class Solver (object):
	"""
	General Solver class, that takes care of calling the step function and storing the intermediate results.

	"""

	def __init__(self, scheme, system, path=None):
		"""
:Parameters:
	system : :class:`odelab.system.System`
		Object describing the system. The requirement on that class may vary depending on the scheme.
	scheme : :class:`odelab.scheme.Scheme`
		Scheme to be used to perform the actual simulation.
	path : :string:
		Path to the file where to save the produced data (if None, a tempfile is created).
		"""
		self.system = system
		self.scheme = scheme
		if path is None: # file does not exist
			import tempfile
			f = tempfile.NamedTemporaryFile(delete=True)
			self.path = f.name
		else:
			self.path = path

	# default values for the total time
	time = 1.

	# default time step
	h = .1



	def initialize(self, u0=None, t0=0, h=None, time=None, name=None):
		"""
Initialize the solver to the initial condition :math:`u(t0) = u0`.

:param array u0: initial condition; if it is not provided, it is set to the previous initial condition.
:param scalar t0: initial time
:param scalar h: time step
:param scalar time: span of the simulation
:param string name: name of this simulation
		"""
		if u0 is None: # initial condition not provided
			raise self.NotInitialized("You must provide an initial condition.")
		if np.isscalar(u0):
			u0 = [u0] # todo: test if this is necessary
		u0 = np.array(u0)
		raw_event0 = np.hstack([u0,t0])
		event0 = self.system.preprocess(raw_event0)

		if h is not None:
			self.h = h
		if time is not None:
			self.time = time

		self.set_name(name=name)

		# first remove the events node if they exist
		#try:
			#self.load_data()
		#except tables.NoSuchNodeError:
			#pass
		#else:
			#self.events.remove()

		# compression algorithm
		compression = tables.Filters(complevel=1, complib='zlib', fletcher32=True)

		with tables.openFile(self.path, 'a') as store:
			# create a new extensible array node
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				events = store.createEArray(
					where=store.root,
					name=self.name,
					atom=tables.Atom.from_dtype(event0.dtype),
					shape=(len(event0),0),
					filters=compression)

			# store the metadata
			info = {
					'u0':u0,
					't0':t0,
					'h':h,
					'time':time,
					}
			events.attrs['init_params'] = info

			# save system and scheme information; temporary solution only
			solver_info = {
				'system': self.system,
				'scheme': self.scheme,
				'solver_class': type(self),
				}
			events.attrs['solver_info'] = solver_info

			# duration counter:
			events.attrs['duration'] = 0.

			# append the initial condition:
			events.append(np.array([event0]).reshape(-1,1)) # todo: factorize the call to reshape, append


	@contextmanager
	def open_store(self, read=True):
		mode = ['a','r'][read]
		with tables.openFile(self.path, mode) as f:
			node = f.getNode('/'+self.name)
			yield node

	def __len__(self):
		with self.open_store(read=True) as events:
			size = events.nrows
		return size

	def get_attrs(self, key):
		with self.open_store() as events:
			attr = events.attrs[key]
		return attr

	def generate(self, event):
		"""
		Generates the (t,u) values.
		"""
		u,t = event[:-1], event[-1]
		for i in itertools.count(): # infinite loop
			t, u = self.step(t, u, self.h)
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

	class RuntimeError(Exception):
		"""
		Raised to relay an exception occurred while running the solver.
		"""

	@contextmanager
	def simulating(self):
		with self.open_store(read=False) as events:

			# only for single step schemes:
			self.set_scheme(self.scheme, events)

			self._start_time = time.time()
			yield events
			end_time = time.time()
			duration = end_time - self._start_time
			events.attrs['duration'] += duration

	auto_save = False # whether to automatically save the session after a run; especially useful for tests


	catch_runtime = True # whether to catch runtime exception (not catching allows to see the traceback)

	t_tol = 1e-12 # tolerance to tell whether the final time is reached

	def run(self, time=None):
		"""
		Run the simulation for a given time.

:param scalar time: the time span for which to run; if none is given, the default ``self.time`` is used
		"""
		if not hasattr(self,'name'):
			raise self.NotInitialized("You must call the `initialize` method before you can run the solver.")
		if time is None:
			time = self.time
		with self.simulating() as events:
			# start from the last time we stopped
			last_event = events[:,-1]
			generator = self.generate(last_event)
			t0 = last_event[-1]
			tf = t0 + time # final time
			for i in xrange(self.max_iter):
				try:
					event = next(generator)
				except Exception as e:
					if self.catch_runtime:
						raise self.RuntimeError('%s raised after %d steps: %s' % (type(e).__name__,i,e.args), e, i)
					else:
						raise
				else:
					if np.any(np.isnan(event)):
						raise self.Unstable('Unstable after %d steps.' % i)

					events.append(event.reshape(-1,1))
					if event[-1] > tf - self.t_tol:
						break
			else:
				raise self.FinalTimeNotReached("Reached maximal number of iterations: {0}".format(self.max_iter))

	def set_name(self, name=None):
		"""
		Set or guess a name for this session.
		"""
		if name is not None:
			self.name = name
		else:
			guess = self.guess_name()
			self.name = guess

	def guess_name(self):
		return "{system}_{scheme}_T{time}".format(system=type(self.system).__name__, scheme=type(self.scheme).__name__, time=self.time,)

	def get_u(self, index, process=True):
		"""
		Return u[index] after post-processing.
		"""
		with self.open_store(read=True) as events:
			event = events[:,index]
		if process:
			event = self.system.postprocess(event)
		return event

	def get_times(self):
		with self.open_store(read=True) as events:
			times = events[-1]
		return times

	def final_time(self):
		with self.open_store(read=True) as events:
			final = events[-1,-1]
		return final

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

	def plot(self, components=None, plot_exact=True, error=False, time_component=None, **plot_args):
		plotter = Plotter(self)
		plotter.setup(plot_exact, error)
		plotter.components = components
		plotter.time_component = time_component
		plotter.plot_args = plot_args
		plotter.plot()
		return plotter

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
		return self.plot(other_component, time_component=time_component, *args, **kwargs)

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

	def __repr__(self):
		return '<{0}: {1} {2}>'.format(type(self).__name__, str(self.scheme), str(self.system))

	def set_scheme(self, scheme, events):
		self.current_scheme = scheme
		self.current_scheme.solver = self
		self.current_scheme.initialize(events)

	def step(self, t,u, h):
		return self.current_scheme.step(t,u,h)

	def increment_stepsize(self):
		self.current_scheme.increment_stepsize()

class MultiStepSolver(SingleStepSolver):
## 	default_single_step_scheme = HochOst4()

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

	def __init__(self, scheme, single_step_scheme=None, *args, **kwargs):
		super(MultiStepSolver, self).__init__(scheme, *args, **kwargs)
		if single_step_scheme is None:
			from odelab.scheme.exponential import HochOst4
			single_step_scheme = HochOst4() # hard coded for the moment
		self.single_step_scheme = single_step_scheme
		self.single_step_scheme.solver = self




def load_solver(path, name):
	"""
Create a solver object from a path to an hdf5 file.
	"""
	with tables.openFile(path, 'r') as f:
		events = f.getNode('/'+name)
		info = events.attrs['solver_info']
		system = info['system']
		scheme = info['scheme']
		solver_class = info['solver_class']
	solver = solver_class(system=system, scheme=scheme, path=path)
	solver.name = name
	return solver
