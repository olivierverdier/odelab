# -*- coding: UTF-8 -*-
from __future__ import division


from odelab.scheme.rungekutta import *
from odelab.scheme import *
from odelab.scheme.classic import *
from odelab.scheme.exponential import *

from odelab.store import Store, PyTableStore, SimpleStore

from odelab.system.classic import *
from odelab.system.exponential import *
from odelab import *
import odelab.newton as rt

import tempfile
import os

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from nose.plugins.skip import SkipTest

import pylab as pl
pl.ioff()

Solver.catch_runtime = False


class Harness(object):
	no_plot = True

def f(t,u):
	return t*np.ones_like(u)

def const_f(c,t,u):
	return c*np.ones_like(u)

def time_f(t,u):
	return t

def test_solver_autosave():
	solver = Solver(ExplicitEuler(h=.1), System(f))
	solver.initialize(u0=1.)
	solver.run()
	nt.assert_equal(solver.guess_name(), 'System_ExplicitEuler_T1.0')

def test_duration():
	"""Duration are added from run to run"""
	solver = Solver(ExplicitEuler(h=.1), System(f))
	solver.initialize(u0=1.,time=1.,)
	solver.run()
	d1 = solver.store['duration']
	solver.run(time=.1)
	d2 = solver.store['duration']
	nt.assert_greater(d2, d1)

def test_initialize_len1():
	solver = Solver(ExplicitEuler(.1),System(f))
	solver.initialize(u0=1.)
	nt.assert_equal(len(solver),1)

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

def test_start_from_two():
	# check that a scheme is not initialized twice, even if we start from more than one event
	dt = .1
	solver = Solver(Scheme_init_once(dt), System(f))
	solver.initialize(u0=1.)
	solver.run(2*dt)
	nt.assert_equal(len(solver),3)
	solver.scheme.is_initialized = False
	solver.run(1.)
	print len(solver)
	print solver.get_events()

@nt.raises(InitializedTwiceError)
def test_initialize_reset_scheme():
	solver = Solver(Scheme_init_once(.1), System(f))
	solver.initialize(u0=1., name='first')
	nt.assert_is(solver.current_scheme, None)
	solver.run(1.)
	solver.initialize(u0=2.,name='second')
	solver.run(1.)

@nt.raises(MultistepInitializationError)
def test_multistep_init_exception():
	multi_scheme = AdamsBashforth(2)
	multi_scheme.h = .1
	s = Solver(scheme=multi_scheme, system=System(f))
	s.initialize(u0=1.)
	with s.open_store() as events:
		s.set_scheme(multi_scheme, events)

class Test_Access(object):
	"""
	Test the Solver.get_events method.
	"""
	def setUp(self):
		self.s = Solver(ExplicitEuler(.1), System(partial(const_f, 1.)))
		self.time = 100
		self.s.initialize(u0=np.array([0.]),time=self.time)
		self.s.run()

	def test_access(self):
		sampling_rate = .5
		evts = self.s.get_events(t0=0, time=50.05, sampling_rate=sampling_rate)
		nt.assert_almost_equal(len(evts.T), len(self.s)*sampling_rate/2, -1) # approx 1/4 of total nb of events
		## nt.assert_equal(len(evts.T), 250)
		npt.assert_array_almost_equal(evts[:,-1], np.array([50.,50.]))


from functools import partial
const_r = partial(const_f, 1.)
const_c = partial(const_f, 1.j)

class Harness_Solver(Harness):
	def setUp(self):
		self.setup_solver()

	dim = 1
	def set_system(self, f):
		self.solver.system = System(f)

	def test_scheme_str(self):
		# should not raise an exception even though h is not yet set in the underlying scheme:
		print str(self.solver)

	def test_initialize(self):
		u0 = np.random.rand(self.dim)
		self.solver.initialize(u0=u0,)
		nt.assert_equal(self.solver.time, Solver.time)
		nt.assert_equal(len(self.solver), 1)

	@nt.raises(PyTableStore.AlreadyInitialized)
	def test_initialize_twice(self):
		if Store is SimpleStore:
			raise SkipTest()
		u0 = np.random.rand(self.dim)
		self.solver.initialize(u0=u0)
		self.solver.initialize(u0=u0)

	def test_initialize_scheme(self):
		raise SkipTest('not relevant anymore, time step is initialized directly at the scheme level')
		h = 10.
		self.solver.initialize(u0=np.random.rand(self.dim),)
		e0 = self.solver.initial()
		with self.solver.open_store() as events:
			self.solver.set_scheme(self.solver.scheme, events)
		self.solver.step(e0[-1], e0[:-1],)
		nt.assert_equal(self.solver.scheme.h, h)

	def test_quadratic(self):
		print type(self).__name__
		self.set_system(time_f)
		self.solver.initialize(u0=1., time=1.,)
		self.solver.run()
		# u'(t) = t; u(0) = u0; => u(t) == u0 + t**2/2
		npt.assert_array_almost_equal(self.solver.final(), np.array([3/2,1.]), decimal=1)

	def check_const(self, f, u0, expected):
		"""should solve the f=c exactly"""
		print type(self).__name__
		self.check_skip(u0,f)
		self.set_system(f)
		self.solver.initialize(u0=u0, time=1.,)
		self.solver.run()
		expected_event = np.hstack([expected, 1.])
		npt.assert_almost_equal(self.solver.final(), expected_event, 1)

	def check_skip(self,u0,f):
		return

	def test_real_const(self):
		self.check_const(const_r, 1., 2.)

	def test_complex_const(self):
		raise SkipTest('Current nonlinear solver does not work with the complex type.')
		self.check_const(const_c, 1.+0j, 1.+1.j)

	def test_repr(self):
		expected = '<Solver: {}'.format(repr(self.solver.scheme))
		r = repr(self.solver)
		nt.assert_true(r.startswith(expected))
		if self.solver.init_scheme is not None:
			nt.assert_regexp_matches(r, repr(self.solver.init_scheme))


class Test_EEuler(Harness_Solver):
	def setup_solver(self):
		self.solver = Solver(ExplicitEuler(h=.1), System(f))

class Test_ETrapezoidal(Harness_Solver):
	def setup_solver(self):
		self.solver = Solver(ExplicitTrapezoidal(h=.1), System(f))

class Test_RK4(Harness_Solver):
	def setup_solver(self):
		self.solver = Solver(RungeKutta4(h=.1), System(f))

class Test_RK34(Harness_Solver):
	def setup_solver(self):
		self.solver = Solver(RungeKutta34(h=.1), System(f))

class Test_AB(Harness_Solver):
	def setup_solver(self):
		multi_scheme = AdamsBashforth(2)
		multi_scheme.h = .1
		self.solver = Solver(multi_scheme, System(f), init_scheme=ExplicitEuler(h=.1))

class Test_RK34Vdp(object):
	def setUp(self):
		time = 7.8
		self.h_init = time/50
		self.scheme = RungeKutta34(h=self.h_init)
		self.s = Solver(self.scheme, VanderPol(mu=1.))
		self.s.initialize(u0 = array([.2,1]), time=time, )

	def test_run(self):
		self.s.run()
		nt.assert_less(self.scheme.h, self.h_init)

class Harness_Solver_NoComplex(Harness_Solver):

	def check_skip(self,u0,f):
		if isinstance(u0,float) and f is const_c:
			raise SkipTest('Does not work with real initial conditions and complex vector fields')

class Test_ode15s(Harness_Solver_NoComplex):
	def setup_solver(self):
		self.solver = Solver(ode15s(h=.1), System(f))

class Test_LawsonEuler(Harness_Solver_NoComplex):
	def set_system(self, f):
		self.solver.system = NoLinear(f,self.dim)
	def setup_solver(self):
		self.solver = Solver(LawsonEuler(h=.1), NoLinear(f,self.dim))

class Test_IEuler(Harness_Solver):
	def setup_solver(self):
		self.solver = Solver(ImplicitEuler(h=.1), System(f))

@nt.raises(Solver.Unstable)
def test_unstable():
	s = Solver(LawsonEuler(h=10.), Linear(np.array([[1.e2]])))
	s.initialize(u0 = 1., time = 100,)
	s.run()

def make_lin(A):
	if np.isscalar(A):
		def lin(t,u):
			return A*u
	else:
		def lin(t, u):
			return dot(A,u)
	lin.exact = make_exp(A)
	return lin

def make_exp(A):
	if np.isscalar(A):
		def exact(u0,t0,t):
			return u0 * exp((t-t0)*A)
	else:
		def exact(u0, t0, t):
			return dot(expm((t-t0)*A),u0)
	return exact

class Harness_Solver_Order(Harness):
	a = -1.
	u0 = 1.
	time = 1.
	do_plot=False
	def notest_order(self):
		self.solver.initialize(u0=self.u0, time=self.time)
		order = self.solver.plot_error(do_plot=self.do_plot)
		print order
		nt.assert_true(order < self.order + .1)

class Test_ExplicitEuler(Harness_Solver_Order):
	def setUp(self):
		self.solver = Solver(ExplicitEuler(h=.1), System(make_lin(self.a)))
		self.order = -1.

class Test_ImplicitEuler(Harness_Solver_Order):
	def setUp(self):
		self.solver = Solver(ImplicitEuler(h=.1), System(make_lin(self.a)))
		self.order = -1.

class Test_RungeKutta4(Harness_Solver_Order):
	def setUp(self):
		self.solver = Solver(RungeKutta4(h=.1), System(make_lin(self.a)))
		self.solver.err_kmin = 1
		self.solver.err_kmax = 2.5
		self.order = -4.





class DummyException(Exception):
	pass

class LimitedSys(System):
	def __init__(self, limit):
		self.limit = limit
		self.i = 0
	def f(self, t, x):
		if self.i < self.limit:
			self.i += 1
			return 0
		else:
			raise DummyException()

class Test_FinalTimeExceptions(object):
	limit = 20
	def setUp(self):
		self.sys = LimitedSys(self.limit)
		self.scheme = ExplicitEuler(h=.1)
		self.s = Solver(self.scheme, self.sys)
		self.s.catch_runtime = True
		self.s.initialize(u0=0, time=10, )

	@nt.raises(Solver.FinalTimeNotReached)
	def test_final_time_not_reached(self):
		self.s.run(max_iter = 1)

	def test_max_iter(self):
		try:
			self.s.run()
		except self.s.RuntimeError:
			pass
		nt.assert_greater_equal(self.s._max_iter, self.s.max_iter_factor*self.s.time/self.scheme.h)
		time = 50
		try:
			self.s.run(50)
		except self.s.RuntimeError:
			pass
		nt.assert_greater_equal(self.s._max_iter, self.s.max_iter_factor*time/self.scheme.h)

	@nt.raises(Solver.RuntimeError)
	def test_sys_exception(self):
		self.s.run()

	@nt.raises(DummyException)
	def test_sys_no_runtime_exception(self):
		self.s.catch_runtime = False
		self.s.run()

def faulty_function(t,u):
	raise Exception('message')

class Test_Exceptions(object):
	def setUp(self):
		self.s = Solver(ExplicitEuler(h=.1), Linear(np.array([[1]])))
	@nt.raises(Solver.NotInitialized)
	def test_no_u0(self):
		self.s.initialize()
	@nt.raises(Solver.NotInitialized)
	def test_no_initialize(self):
		self.s.run()
	@nt.raises(Solver.Unstable)
	def test_unstable(self):
		self.s = Solver(ExplicitEuler(h=.1), Linear(np.array([[float('inf')]])))
		self.s.initialize(u0=np.array([0]))
		self.s.run()
	@nt.raises(Solver.RuntimeError)
	def test_runtime_exception(self):
		self.s = Solver(ExplicitEuler(h=.1), System(faulty_function))
		self.s.catch_runtime = True
		self.s.initialize(u0=0)
		self.s.run()

class TotSys(System):
	def total(self, xt):
		return np.sum(xt[:-1],axis=0)

def minus_x(t, x):
	return -x

class Test_Simple(object):
	def setUp(self):
		sys = TotSys(minus_x)
		self.s = Solver(ExplicitEuler(h=.1), sys)

	def test_time(self):
		sol = self.s
		sol.h = Solver.time/10
		sol.initialize(u0=0.)
		sol.run(sol.h)
		npt.assert_(sol.final_time() < Solver.time)

	def test_extra_run(self):
		"""test that an extra run continues from last time"""
		sol = self.s
		sol.initialize(u0=1.)
		sol.run()
		npt.assert_almost_equal(sol.final_time(),Solver.time)
		sol.run()
		npt.assert_almost_equal(sol.final_time(),2*Solver.time)

	def test_plot_args(self):
		self.s.initialize(u0=np.array([1.,1.,1.]))
		self.s.run()
		pl.clf()
		lines = self.s.plot(0,lw=5).axis.lines
		npt.assert_equal(len(lines),1)
		pl.clf()
		lines = self.s.plot(lw=5).axis.lines
		npt.assert_equal(len(lines),3)
		npt.assert_equal(lines[-1].get_linewidth(),5)

	def test_plot_function(self):
		self.s.initialize(u0=np.array([1.,1.,1.]))
		self.s.run()
		lines = self.s.plot_function('total', lw=4).axis.lines
		npt.assert_equal(lines[-1].get_linewidth(), 4)


pl.ion()

