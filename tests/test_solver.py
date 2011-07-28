# -*- coding: UTF-8 -*-
from __future__ import division


from odelab.scheme import *
from odelab.scheme.exponential import *

from odelab.system import *
from odelab.solver import *
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
	solver = SingleStepSolver(ExplicitEuler(h=.1), System(f))
	solver.initialize(u0=1.)
	solver.run()
	nt.assert_equal(solver.guess_name(), 'System_ExplicitEuler_T1.0')

def test_duration():
	"""Duration are added from run to run"""
	solver = SingleStepSolver(ExplicitEuler(h=.1), System(f))
	solver.initialize(u0=1.,time=1.,)
	solver.run()
	d1 = solver.get_attrs('duration')
	solver.run(time=.1)
	d2 = solver.get_attrs('duration')
	nt.assert_greater(d2, d1)


from functools import partial
const_r = partial(const_f, 1.)
const_c = partial(const_f, 1.j)

class Harness_Solver(Harness):
	def setUp(self):
		self.setup_solver()
		self.solver.auto_save = False

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

	def test_initialize_twice(self):
		raise SkipTest('double initialization is deactivated')
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
		self.solver.scheme.root_solver = rt.Newton
		self.solver.run()
		expected_event = np.hstack([expected, 1.])
		npt.assert_almost_equal(self.solver.final(), expected_event, 1)

	def check_skip(self,u0,f):
		return

	def test_const(self):
		for f,u0_,expected in [(const_r, 1., 2.), (const_c, 1.+0j, 1.+1.j),]:
			yield self.check_const, f, u0_, expected

class Test_EEuler(Harness_Solver):
	def setup_solver(self):
		self.solver = SingleStepSolver(ExplicitEuler(h=.1), System(f))

class Test_ETrapezoidal(Harness_Solver):
	def setup_solver(self):
		self.solver = SingleStepSolver(ExplicitTrapezoidal(h=.1), System(f))

class Test_RK4(Harness_Solver):
	def setup_solver(self):
		self.solver = SingleStepSolver(RungeKutta4(h=.1), System(f))

class Test_RK34(Harness_Solver):
	def setup_solver(self):
		self.solver = SingleStepSolver(RungeKutta34(h=.1), System(f))

class Harness_Solver_NoComplex(Harness_Solver):

	def check_skip(self,u0,f):
		if isinstance(u0,float) and f is const_c:
			raise SkipTest('Does not work with real initial conditions and complex vector fields')

class Test_ode15s(Harness_Solver_NoComplex):
	def setup_solver(self):
		self.solver = SingleStepSolver(ode15s(h=.1), System(f))

class Test_LawsonEuler(Harness_Solver_NoComplex):
	def set_system(self, f):
		self.solver.system = NoLinear(f,self.dim)
	def setup_solver(self):
		self.solver = SingleStepSolver(LawsonEuler(h=.1), NoLinear(f,self.dim))

class Test_IEuler(Harness_Solver):
	def setup_solver(self):
		self.solver = SingleStepSolver(ImplicitEuler(h=.1), System(f))

@nt.raises(Solver.Unstable)
def test_unstable():
	s = SingleStepSolver(LawsonEuler(h=10.), Linear(np.array([[1.e2]])))
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
		self.solver = SingleStepSolver(ExplicitEuler(h=.1), System(make_lin(self.a)))
		self.order = -1.

class Test_ImplicitEuler(Harness_Solver_Order):
	def setUp(self):
		self.solver = SingleStepSolver(ImplicitEuler(h=.1), System(make_lin(self.a)))
		self.order = -1.

class Test_RungeKutta4(Harness_Solver_Order):
	def setUp(self):
		self.solver = SingleStepSolver(RungeKutta4(h=.1), System(make_lin(self.a)))
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
		self.s = SingleStepSolver(ExplicitEuler(h=.1),self.sys)
		self.s.catch_runtime = True
		self.s.initialize(u0=0, time=10, )

	def test_max_iter(self):
		self.max_iter = 1
		self.s.max_iter = self.max_iter
		try:
			self.s.run()
		except Solver.FinalTimeNotReached:
			npt.assert_equal(len(self.s), self.max_iter + 1)
		else:
			raise Exception("FinalTimeNotReached not raised!")

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
		self.s = SingleStepSolver(ExplicitEuler(h=.1), Linear(np.array([[1]])))
		self.s.auto_save = False
	@nt.raises(Solver.NotInitialized)
	def test_no_u0(self):
		self.s.initialize()
	@nt.raises(Solver.NotInitialized)
	def test_no_initialize(self):
		self.s.run()
	@nt.raises(Solver.Unstable)
	def test_unstable(self):
		self.s = SingleStepSolver(ExplicitEuler(h=.1), Linear(np.array([[float('inf')]])))
		self.s.initialize(u0=np.array([0]))
		self.s.run()
	@nt.raises(Solver.RuntimeError)
	def test_runtime_exception(self):
		self.s = SingleStepSolver(ExplicitEuler(h=.1), System(faulty_function))
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
		self.s = SingleStepSolver(ExplicitEuler(h=.1), sys)

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

