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

import itertools
import warnings
import time

import progressbar as pb
widgets = ['',' ', pb.Timer('%s'),' ', pb.Percentage(), ' ', pb.Bar(u'■'), ' ',  ' ', pb.ETA(),  ]
progress_bar = pb.ProgressBar(widgets=widgets)
del pb


from odelab.plotter import Plotter

import tables

from contextlib import contextmanager

class Solver (object):
	"""
	General Solver class, that takes care of calling the step function and storing the intermediate results.

	"""

	def __init__(self, scheme, system, path=None, init_scheme=None):
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
		self.init_scheme = init_scheme
		if path is None: # file does not exist
			import tempfile
			f = tempfile.NamedTemporaryFile(delete=True)
			self.path = f.name
		else:
			self.path = path

	# default values for the total time
	time = 1.

	# max_iter = max_iter_factor * (time/h)
	max_iter_factor = 100

	def initialize(self, u0=None, t0=0, time=None, name=None):
		"""
Initialize the solver to the initial condition :math:`u(t0) = u0`.

:param array u0: initial condition; if it is not provided, it is set to the previous initial condition.
:param scalar t0: initial time
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

		if time is not None:
			self.time = time

		self.set_name(name=name)

		# compression algorithm
		compression = tables.Filters(complevel=1, complib='zlib', fletcher32=True)

		with tables.openFile(self.path, 'a') as store:
			# create a new extensible array node
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				try:
					events = store.createEArray(
						where=store.root,
						name=self.name,
						atom=tables.Atom.from_dtype(event0.dtype),
						shape=(len(event0),0),
						filters=compression)
				except tables.NodeError:
					raise self.AlreadyInitialized('Results with the name "{0}" already exist in that store.'.format(self.name))

			# store the metadata
			info = {
					'u0':u0,
					't0':t0,
					'time':time,
					}
			events.attrs['init_params'] = info

			# save system and scheme information; temporary solution only
			solver_info = {
				'system': self.system,
				'scheme': self.scheme,
				'init_scheme': self.init_scheme,
				'solver_class': type(self),
				}
			events.attrs['solver_info'] = solver_info
			events.attrs['solver'] = self

			# duration counter:
			events.attrs['duration'] = 0.

			# append the initial condition:
			events.append(np.array([event0]).reshape(-1,1)) # todo: factorize the call to reshape, append


	@contextmanager
	def open_store(self, write=False):
		"""
Method to open the data store. Any access to the events must make use of this method::

	with solver.open_store() as events:
		...
		"""
		mode = ['r','a'][write]
		with tables.openFile(self.path, mode) as f:
			node = f.getNode('/'+self.name)
			yield node

	def __len__(self):
		with self.open_store() as events:
			size = events.nrows
		return size

	def get_attrs(self, key):
		with self.open_store() as events:
			attr = events.attrs[key]
		return attr

	def generate(self, events):
		"""
		Generates the (t,u) values.
		"""
		last_event = events[:,-1]
		event = last_event
		init_stage = len(events)
		tail_length = self.scheme.tail_length
		for stage in itertools.count(init_stage): # infinite loop
			if stage < tail_length: # not enough past values to run main scheme
				if stage == 1:
					self.set_scheme(self.init_scheme, events)
			if stage == tail_length or init_stage > tail_length: # main scheme kicks in
				self.set_scheme(self.scheme, events)
			event = self.step(event)
			yield event


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

	class AlreadyInitialized(Exception):
		"""
		Raised when a solver is already initialized.
		"""

	class RuntimeError(Exception):
		"""
		Raised to relay an exception occurred while running the solver.
		"""

	@contextmanager
	def simulating(self):
		with self.open_store(write=True) as events:
			self._start_time = time.time()
			yield events
			end_time = time.time()
			duration = end_time - self._start_time
			events.attrs['duration'] += duration

	catch_runtime = True # whether to catch runtime exception (not catching allows to see the traceback)

	t_tol = 1e-12 # tolerance to tell whether the final time is reached

	def run(self, time=None, max_iter=None):
		"""
		Run the simulation for a given time.

:param scalar time: the time span for which to run; if none is given, the default ``self.time`` is used
:param max_iter: the maximum number of iterations; if None, an estimate is computed base on the time step and time span
		"""
		if not hasattr(self,'name'):
			raise self.NotInitialized("You must call the `initialize` method before you can run the solver.")
		if time is None:
			time = self.time

		self._max_iter = max_iter
		if self._max_iter is None:
			# generous estimation of the maximum number of iterations
			self._max_iter = int(time/self.scheme.h * self.max_iter_factor)

		with self.simulating() as events:
			# start from the last time we stopped
			generator = self.generate(events)
			t0 = events[-1,-1]
			tf = t0 + time # final time

			progress_bar.maxval = time
			progress_bar.widgets[0] = self.name
			progress_bar.start()

			for iteration in xrange(self._max_iter): # todo: use enumerate
				try:
					event = next(generator)
				except Exception as e:
					if self.catch_runtime:
						raise self.RuntimeError('%s raised after %d steps: %s' % (type(e).__name__,iteration,e.args), e, iteration)
					else:
						raise
				else:
					if np.any(np.isnan(event)):
						raise self.Unstable('Unstable after %d steps.' % iteration)

					events.append(event.reshape(-1,1))
					t = event[-1]
					progress_bar.update(t-t0)
					if t > tf - self.t_tol:
						break
			else:
				raise self.FinalTimeNotReached("Reached maximal number of iterations: {0}".format(self._max_iter))

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
		with self.open_store() as events:
			event = events[:,index]
		if process:
			event = self.system.postprocess(event)
		return event

	def get_times(self):
		with self.open_store() as events:
			times = events[-1]
		return times

	def final_time(self):
		with self.open_store() as events:
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

	def __repr__(self):
		solver = type(self).__name__
		scheme = repr(self.scheme)
		if self.init_scheme is not None:
			init_scheme = '({})'.format(repr(self.init_scheme))
		else:
			init_scheme = ''
		system = repr(self.system)
		return '<{solver}: {scheme}{init_scheme} {system}>'.format(solver=solver, scheme=scheme, init_scheme=init_scheme, system=system)

	def set_scheme(self, scheme, events):
		self.current_scheme = scheme
		self.current_scheme.system = self.system
		self.current_scheme.initialize(events)

	def step(self, event):
		return self.current_scheme.do_step()

SingleStepSolver = Solver



def load_solver(path, name):
	"""
Create a solver object from a path to an hdf5 file.
	"""
	with tables.openFile(path, 'r') as f:
		events = f.getNode('/'+name)
		try:
			solver = events.attrs['solver']
		except KeyError:
			solver = load_solver_v2(path, name)
	return solver

def load_solver_v2(path, name):
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
