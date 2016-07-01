# -*- coding: utf-8 -*-
"""
:mod:`Solver` -- ODE Solvers
============================

.. module :: solver
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

The class :class:`~odelab.solver.Solver` takes care of calling the numerical scheme to produce data, and of storing that data.


The higher level class is :class:`Solver`, which is initialized with an instance of a :class:`Scheme` class.
"""
from __future__ import division

import numpy as np

import itertools
import time



from odelab.plotter import Plotter

import warnings
from .store import Store, SimpleStore

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
		Path to the file where to save the produced data (if ``None``, a tempfile is created).
		"""
		self.system = system
		self.scheme = scheme
		self.init_scheme = init_scheme
		if path is None:
			self.store = SimpleStore()
		else:
			self.store = Store(path)

	# max_iter = max_iter_factor * (time/h)
	max_iter_factor = 100

	def initialize(self, u0=None, t0=0, name=None):
		"""
Initialize the solver to the initial condition :math:`u(t0) = u0`.

:param array u0: initial condition; if it is not provided, it is set to the previous initial condition.
:param scalar time: span of the simulation
:param string name: name of this simulation
		"""
		self.current_scheme = None
		if u0 is None: # initial condition not provided
			raise self.NotInitialized("You must provide an initial condition.")
		if np.isscalar(u0):
			u0 = [u0] # todo: test if this is necessary
		u0 = np.array(u0)
		raw_event0 = np.hstack([u0, t0])
		event0 = self.system.preprocess(raw_event0)

		self.set_name(name=name)

		info = {
				'u0':u0,
				't0':t0,
				}
		# save system and scheme information in order to recover if unpickling fails
		solver_info = {
			'system_class': repr(type(self.system)),
			'scheme_class': repr(type(self.scheme)),
			'init_scheme_class': repr(type(self.init_scheme)),
			'solver_class': repr(type(self)),
			}

		self.store.initialize(event0, self.name)

		with self.open_store(write=True):

			# store the metadata
			self.store['init_params'] = info

			self.store['solver_info'] = solver_info
			self.store['solver'] = self

			# duration counter:
			self.store['duration'] = 0.

			# append the initial condition:
			self.store.append(event0)


	@contextmanager
	def open_store(self, write=False):
		"""
Method to open the data store. Any access to the events must make use of this method::

	with solver.open_store() as events:
		...
		"""
		with self.store.open(write) as events:
			yield events

	def __len__(self):
		return len(self.store)

	def generate(self, events):
		"""
		Generates the (t,u) values.
		"""
		last_event = events[:, -1]
		event = last_event
		init_stage = self.store.get_nb_stage(events)
		tail_length = self.scheme.tail_length
		for stage in itertools.count(init_stage): # infinite loop
			if stage < tail_length: # not enough past values to run main scheme
				if stage == 1:
					self.set_scheme(self.init_scheme, events)
			elif self.current_scheme is None: # main scheme kicks in
				self.set_scheme(self.scheme, events)
			event = self.step()
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


	class RuntimeError(Exception):
		"""
		Raised to relay an exception occurred while running the solver.
		"""

	class NotRun(Exception):
		"""
		Raised when trying to access the events although the solver is empty.
		"""

	@contextmanager
	def simulating(self):
		self._start_time = time.time()
		with self.open_store(write=True) as events:
			yield events
			end_time = time.time()
			duration = end_time - self._start_time
			self.store['duration'] += duration

	catch_runtime = True # whether to catch runtime exception (not catching allows to see the traceback)

	t_tol = 1e-12 # tolerance to tell whether the final time is reached

	def run(self, time, max_iter=None):
		"""
		Run the simulation for a given time.

:param scalar time: the time span for which to run;
:param max_iter: the maximum number of iterations; if ``None``, an estimate is computed base on the time step and time span
		"""
		if not hasattr(self,'name'):
			raise self.NotInitialized("You must call the `initialize` method before you can run the solver.")

		self._max_iter = max_iter
		if self._max_iter is None:
			# generous estimation of the maximum number of iterations
			self._max_iter = int(time/self.scheme.h * self.max_iter_factor)



		with self.simulating() as events:
			# start from the last time we stopped
			generator = self.generate(events)
			t0 = events[-1, -1]
			tf = t0 + time # final time

			if self.with_progressbar:
				progress_bar.maxval = time
				progress_bar.widgets[0] = self.name
				progress_bar.start()

			for iteration in range(self._max_iter): # todo: use enumerate
				try:
					event = next(generator)
				except Exception as e:
					if self.catch_runtime:
						raise self.RuntimeError('%s raised after %d steps: %s' % (type(e).__name__, iteration, e.args), e, iteration)
					else:
						raise
				else:
					if np.any(np.isnan(event)):
						raise self.Unstable('Unstable after %d steps.' % iteration)

					self.store.append(event)
					t = event[-1]
					if self.with_progressbar:
						progress_bar.update(np.real(t-t0))
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
		return "{system}_{scheme}".format(system=type(self.system).__name__, scheme=type(self.scheme).__name__, )

	def get_u(self, index, process=True):
		"""
		Return u[index] after post-processing.
		"""
		with self.open_store() as events:
			event = events[:, index]
		if process:
			event = self.system.postprocess(event)
		return event

	def get_events(self, t0=None, time=None, sampling_rate=1.):
		"""
		Return the events from time t0, during time `time`, sampled.
		"""
		ts = self.get_times()
		if len(ts) == 1:
			raise self.NotRun('The solver has not been run. If you wanted to check the initial condition, use Solver.initial().')
		if t0 is None:
			t0 = ts[0]
		if time is None:
			time = ts[-1] - ts[0]
		indices = np.where((ts >= t0) & (ts < t0 + time))[0]
		size = len(indices)
		initial_index = indices[0]
		final_index = indices[-1]+1
		stride = int(np.ceil(1/sampling_rate))
		with self.open_store() as events:
			result = events[:, slice(initial_index, final_index, stride)]
			return result

	def get_times(self):
		with self.open_store() as events:
			times = events[-1]
		return times

	def final_time(self):
		with self.open_store() as events:
			final = events[-1, -1]
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

	def plot(self, *args, **kwargs):
		"""
Plot using the plotter object from :method:`odelab.Solver.plotter`.
		"""
		plotter = self.plotter(*args, **kwargs)
		plotter.plot()
		return plotter

	def plotter(self, components=None, plot_exact=True, error=False, time_component=None, t0=None, time=None, **plot_args):
		"""
Constructs a plotter object.
		"""
		plotter = Plotter(self)
		plotter.setup(plot_exact, error)
		plotter.components = components
		plotter.time_component = time_component
		plotter.plot_args = plot_args
		plotter.t0 = t0
		plotter.time = time
		return plotter

	def plot_function(self, function, *args, **kwargs):
		"""
		Plot a given function of the state. May be useful to plot constraints or energy.

		This is now a convenience function that calls the method :meth:`plot`.

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
			init_scheme = '({0})'.format(repr(self.init_scheme))
		else:
			init_scheme = ''
		system = repr(self.system)
		return '<{solver}: {scheme}{init_scheme} {system}>'.format(solver=solver, scheme=scheme, init_scheme=init_scheme, system=system)

	def set_scheme(self, scheme, events):
		self.current_scheme = scheme
		self.current_scheme.system = self.system
		self.current_scheme.initialize(events)

	def step(self):
		return self.current_scheme.do_step()

SingleStepSolver = Solver



def load_solver(path, name):
	"""
Create a solver object from a path to an hdf5 file.
	"""
	import tables
	with tables.openFile(path, 'r') as f:
		events = f.getNode('/'+name)
		try:
			solver = events.attrs['solver']
		except KeyError:
			solver = load_solver_v2(path, name)
		if not isinstance(solver, Solver): # pickling has failed
			warnings.warn('Loading failed')
			solver = Solver(scheme=None, system=None, path=path)
			solver.name = name
		solver.store = Store(path)
		solver.store.name = solver.name
	return solver

def load_solver_v2(path, name):
	"""
Create a solver object from a path to an hdf5 file.
	"""
	import tables
	with tables.openFile(path, 'r') as f:
		events = f.getNode('/'+name)
		info = events.attrs['solver_info']
		if isinstance(info, dict):
			system = info['system']
			scheme = info['scheme']
			solver_class = info['solver_class']
			solver = solver_class(system=system, scheme=scheme, path=path)
		else:
			solver = Solver(scheme=None, system=None, path=path)
		solver.name = name
	return solver

# try to import progressbar and use it if it is available
try:
	import progressbar as pb
	widgets = ['', ' ', pb.Timer('%s'), ' ', pb.Percentage(), ' ', pb.Bar('='), ' ',  ' ', pb.ETA(),  ]
	progress_bar = pb.ProgressBar(widgets=widgets)
	del pb
	with_progressbar = True
except ImportError:
	with_progressbar = False

Solver.with_progressbar = with_progressbar
