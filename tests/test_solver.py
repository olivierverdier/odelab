# -*- coding: utf-8 -*-
from __future__ import division

from odelab.scheme.rungekutta import *
from odelab.scheme.generallinear import *
from odelab.scheme import *
from odelab.scheme.classic import *
from odelab.scheme.exponential import *

from odelab.store import Store, PyTableStore, SimpleStore

from odelab.system.classic import *
from odelab.system.exponential import *
from odelab import *

import tempfile
import os

import numpy as np
import numpy.testing as npt
import unittest
import pytest

import matplotlib.pyplot as plt

Solver.catch_runtime = False



def f(t,u):
	return t*np.ones_like(u)


class TestMisc(unittest.TestCase):
	def test_solver_autosave(self):
		solver = Solver(ExplicitEuler(h=.1), System(f))
		solver.initialize(u0=1.)
		solver.run(1.)
		self.assertEqual(solver.guess_name(), 'System_ExplicitEuler')

	def test_duration(self):
		"""Duration are added from run to run"""
		solver = Solver(ExplicitEuler(h=.1), System(f))
		solver.initialize(u0=1.,)
		solver.run(1.)
		d1 = solver.store['duration']
		solver.run(time=.1)
		d2 = solver.store['duration']
		self.assertGreater(d2, d1)


class InitializedTwiceError(ValueError):
	pass

class Scheme_init_once(ExplicitEuler):
	def __init__(self, *args,**kwargs):
		super(Scheme_init_once,self).__init__(*args, **kwargs)
		self.is_initialized = False

	def initialize(self, events):
		if self.is_initialized:
			raise InitializedTwiceError('initialized twice!')
		super(Scheme_init_once,self).initialize(events)
		self.is_initialized = True

class TestInitialize(unittest.TestCase):
	def test_initialize_len1(self):
		solver = Solver(ExplicitEuler(.1),System(f))
		solver.initialize(u0=1.)
		self.assertEqual(len(solver),1)

	def test_start_from_two(self):
		# check that a scheme is not initialized twice, even if we start from more than one event
		dt = .1
		solver = Solver(Scheme_init_once(dt), System(f))
		solver.initialize(u0=1.)
		solver.run(2*dt)
		self.assertEqual(len(solver),3)
		solver.scheme.is_initialized = False
		solver.run(1.)
		print(len(solver))
		print(solver.get_events())

	def test_initialize_reset_scheme(self):
		solver = Solver(Scheme_init_once(.1), System(f))
		solver.initialize(u0=1., name='first')
		self.assertIs(solver.current_scheme, None)
		solver.run(1.)
		solver.initialize(u0=2.,name='second')
		with self.assertRaises(InitializedTwiceError):
			solver.run(1.)

	def test_multistep_init_exception(self):
		multi_scheme = AdamsBashforth2(.1)
		s = Solver(scheme=multi_scheme, system=System(f))
		s.initialize(u0=1.)
		with self.assertRaises(MultistepInitializationError):
			with s.open_store() as events:
				s.set_scheme(multi_scheme, events)

class Test_Access(unittest.TestCase):
	"""
	Test the Solver.get_events method.
	"""
	def setUp(self):
		self.s = Solver(ExplicitEuler(.1), System(f))
		self.time = 30
		self.s.initialize(u0=np.array([0.]))

	def test_access(self):
		self.s.run(self.time)
		sampling_rate = .5
		time = 15
		evts = self.s.get_events(t0=0, time=time, sampling_rate=sampling_rate)
		npt.assert_almost_equal(len(evts.T), len(self.s)*sampling_rate/2, -1) # approx 1/4 of total nb of events
		## self.assertEqual(len(evts.T), 250)

	def test_notrun(self):
		with self.assertRaises(Solver.NotRun):
			self.s.get_events()



class TestSolver(unittest.TestCase):
	def setUp(self):
		self.setup_solver()

	def setup_solver(self):
		self.solver = Solver(ExplicitEuler(h=.1), System(f))
		self.solver.store = Store(None)

	dim = 1
	def set_system(self, f):
		self.solver.system = System(f)

	def test_scheme_str(self):
		# should not raise an exception even though h is not yet set in the underlying scheme:
		print(str(self.solver))

	def test_initialize(self):
		u0 = np.random.rand(self.dim)
		self.solver.initialize(u0=u0,)
		self.assertEqual(len(self.solver), 1)

	def test_initialize_twice(self):
		if isinstance(self.solver.store, SimpleStore):
			pytest.skip()
		u0 = np.random.rand(self.dim)
		self.solver.initialize(u0=u0)
		with self.assertRaises(PyTableStore.AlreadyInitialized):
			self.solver.initialize(u0=u0)

	@pytest.mark.skip('Not relevant anymore, time step is initialized directly at the scheme level')
	def test_initialize_scheme(self):
		h = 10.
		self.solver.initialize(u0=np.random.rand(self.dim),)
		e0 = self.solver.initial()
		with self.solver.open_store() as events:
			self.solver.set_scheme(self.solver.scheme, events)
		self.solver.step(e0[-1], e0[:-1],)
		self.assertEqual(self.solver.scheme.h, h)




class Test_RK34Vdp(unittest.TestCase):
	def setUp(self):
		self.time = 7.8
		self.h_init = self.time/50
		self.scheme = RungeKutta34(h=self.h_init)
		self.s = Solver(self.scheme, VanderPol(mu=1.))
		self.s.initialize(u0 = array([.2,1]))

	def test_run(self):
		self.s.run(self.time)
		self.assertLess(self.scheme.h, self.h_init)

class TestStability(unittest.TestCase):
	def test_unstable(self):
		with self.assertRaises(Solver.Unstable):
			s = Solver(LawsonEuler(h=10.), Linear(np.array([[1.e2]])))
			s.initialize(u0 = 1.)
			s.run(100)


class DummyException(Exception):
	pass

class LimitedSys(System):
	def __init__(self, limit):
		self.limit = limit
		self.i = 0
	def __call__(self, t, x):
		if self.i < self.limit:
			self.i += 1
			return 0
		else:
			raise DummyException()

class Test_FinalTimeExceptions(unittest.TestCase):
	limit = 20
	def setUp(self):
		self.sys = LimitedSys(self.limit)
		self.scheme = ExplicitEuler(h=.1)
		self.s = Solver(self.scheme, self.sys)
		self.s.catch_runtime = True
		self.time = 10
		self.s.initialize(u0=0)

	def test_final_time_not_reached(self):
		with self.assertRaises(Solver.FinalTimeNotReached):
			self.s.run(self.time, max_iter = 1)

	def test_max_iter(self):
		try:
			self.s.run(self.time)
		except self.s.RuntimeError:
			pass
		self.assertGreaterEqual(self.s._max_iter, self.s.max_iter_factor*self.time/self.scheme.h)
		time = 50
		try:
			self.s.run(50)
		except self.s.RuntimeError:
			pass
		self.assertGreaterEqual(self.s._max_iter, self.s.max_iter_factor*time/self.scheme.h)

	def test_sys_exception(self):
		with self.assertRaises(Solver.RuntimeError):
			self.s.run(self.time)

	def test_sys_no_runtime_exception(self):
		self.s.catch_runtime = False
		with self.assertRaises(DummyException):
			self.s.run(self.time)

def faulty_function(t,u):
	raise Exception('message')

class Test_Exceptions(unittest.TestCase):
	def setUp(self):
		self.s = Solver(ExplicitEuler(h=.1), Linear(np.array([[1]])))
	def test_no_u0(self):
		with self.assertRaises(Solver.NotInitialized):
			self.s.initialize()
	def test_no_initialize(self):
		with self.assertRaises(Solver.NotInitialized):
			self.s.run(1.)
	def test_unstable(self):
		self.s = Solver(ExplicitEuler(h=.1), Linear(np.array([[float('inf')]])))
		self.s.initialize(u0=np.array([0]))
		with self.assertRaises(Solver.Unstable):
			self.s.run(1.)
	def test_runtime_exception(self):
		self.s = Solver(ExplicitEuler(h=.1), System(faulty_function))
		self.s.catch_runtime = True
		self.s.initialize(u0=0)
		with self.assertRaises(Solver.RuntimeError):
			self.s.run(1.)

class TotSys(System):
	def total(self, xt):
		return np.sum(xt[:-1],axis=0)

def minus_x(t, x):
	return -x

class Test_Simple(unittest.TestCase):
	def setUp(self):
		sys = TotSys(minus_x)
		self.s = Solver(ExplicitEuler(h=.1), sys)

	def test_time(self):
		sol = self.s
		time = 1.
		sol.h = time/10
		sol.initialize(u0=0.)
		sol.run(sol.h)
		npt.assert_(sol.final_time() < time)

	def test_extra_run(self):
		"""test that an extra run continues from last time"""
		sol = self.s
		sol.initialize(u0=1.)
		time = 1.
		sol.run(1.)
		npt.assert_almost_equal(sol.final_time(), time)
		sol.run(time)
		npt.assert_almost_equal(sol.final_time(),2*time)

	def test_plot_args(self):
		self.s.initialize(u0=np.array([1.,1.,1.]))
		self.s.run(1.)
		plt.clf()
		lines = self.s.plot(0,lw=5).axis.lines
		npt.assert_equal(len(lines),1)
		plt.clf()
		lines = self.s.plot(lw=5).axis.lines
		npt.assert_equal(len(lines),3)
		npt.assert_equal(lines[-1].get_linewidth(),5)

	def test_plot_function(self):
		self.s.initialize(u0=np.array([1.,1.,1.]))
		self.s.run(1.)
		lines = self.s.plot_function('total', lw=4).axis.lines
		npt.assert_equal(lines[-1].get_linewidth(), 4)



