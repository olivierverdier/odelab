# -*- coding: UTF-8 -*-
from __future__ import division


from odelab.solver import *
from odelab.system import *

import numpy.testing as npt
import nose.tools as nt

class Harness(object):
	no_plot = True

def f(t,u):
	return t

def res(u, t, h):
	return u + (t+h)*h

f.res = res

def const_f(t,u):
	return 1.

def time_f(t,u):
	return t

class Harness_Solver(Harness):
	def test_initialize(self):
		u0 = np.random.rand(5)
		self.solver.initialize(u0=u0)
		nt.assert_true(self.solver.time == Solver.time)
		nt.assert_true(self.solver.scheme.h == Scheme.h_default)
		nt.assert_true(len(self.solver.ts) == 1)
		nt.assert_true(len(self.solver.us) == 1)
	
	def test_quadratic(self):
		"""should solve f(t) = t pretty well"""
		print type(self).__name__
		self.solver.system = System(time_f)
		self.solver.initialize(u0=1., time=1.)
		self.solver.run()
		# u'(t) = t; u(0) = u0; => u(t) == u0 + t**2/2
		nt.assert_almost_equal(self.solver.us[-1], 3/2, 1)

	def test_const(self):
		"""should solve the f=1. exactly"""
		print type(self).__name__
		self.solver.system = System(const_f)
		self.solver.initialize(u0=1., time=1.)
		self.solver.run()
		npt.assert_almost_equal(self.solver.us[-1], 2., 1)

class Test_EEuler(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(ExplicitEuler(), System(f))

class Test_RK4(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(RungeKutta4(), System(f))

class Test_RK34(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(RungeKutta34(), System(f))

## class Test_IEuler(Harness_Solver):
## 	def setUp(self):
## 		self.solver = SingleStepSolver(ImplicitEuler, System(f))

class Harness_Circle(Harness):
	def setUp(self):
		def f(t,u):
			return array([-u[1], u[0]])
		self.f = f
		self.make_solver()
		self.s.initialize(u0 = array([1.,0.]), h=.01, time = 10.)
	
	def run(self):
		self.s.run()
		self.s.plot2D()

class Test_Circle_EEuler(Harness_Circle):
	def make_solver(self):
		self.s = ExplicitEuler(self.f)

class Test_Circle_RK34(Harness_Circle):
	def make_solver(self):
		self.s = RungeKutta34(self.f)

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

class Harness_Solver(Harness):
	a = -1.
	u0 = 1.
	time = 1.
	do_plot=False
	def notest_order(self):
		self.solver.initialize(u0=self.u0, time=self.time)
		order = self.solver.plot_error(do_plot=self.do_plot)
		print order
		nt.assert_true(order < self.order + .1)
	
class Test_ExplicitEuler(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(ExplicitEuler(), System(make_lin(self.a)))
		self.order = -1.

class Test_RungeKutta4(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(RungeKutta4(), System(make_lin(self.a)))
		self.solver.err_kmin = 1
		self.solver.err_kmax = 2.5
		self.order = -4.


class Harness_Osc(object):
	def setUp(self):
		self.sys = ContactOscillator()
		self.set_solver()
		self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
		self.s.time = 10.
	
## 	def test_run(self):
## 		self.s.run()
	
	
	z0s = np.linspace(-.9,.9,10)
	N = 40
	
	def test_z0(self, i=5, nb_Poincare_iterations=10):
		z0 = self.z0s[i]
		self.s.initialize(u0=self.sys.initial(z0))
		self.s.scheme.h = self.sys.time_step(self.N)
		self.s.time = nb_Poincare_iterations*self.N*self.s.scheme.h
		self.s.run()
		npt.assert_almost_equal(self.sys.energy(self.s.us[-1]), self.sys.energy(self.s.us[0]), decimal=1)

	def plot_qv(self, i=2, skip=None, *args, **kwargs):
		if skip is None:
			skip = self.N
		qs = self.sys.position(self.s.aus)
		vs = self.sys.velocity(self.s.aus)
		if not kwargs.get('marker') and not kwargs.get('ls'):
			kwargs['ls'] = ''
			kwargs['marker'] = 'o'
		plot(qs[i,::skip], vs[i,::skip], *args, **kwargs)
	
		
class Test_McOsc(Harness_Osc):
	def set_solver(self):
		self.s = SingleStepSolver(McLachlan(), self.sys)

class Test_JayOsc(Harness_Osc):
	def set_solver(self):
		self.s = SingleStepSolver(Spark(2), self.sys)
	
class Test_SparkODE(object):
	def setUp(self):
		def f(xt):
			return -xt[0]
		self.sys = ODESystem(f)
		self.s = SingleStepSolver(Spark(4), self.sys)
		self.s.initialize(array([1.]))
	
	def test_run(self):
		self.s.run()
		exact = np.exp(-self.s.ats)
		print exact[-1]
		print self.s.us[-1]
## 		npt.assert_array_almost_equal(self.s.aus, exact, 5)
		npt.assert_almost_equal(self.s.us[-1], exact[-1])
## 		plot(self.s.ats, np.vstack([self.s.aus, exact]).T)

class Test_Jay(object):
	def setUp(self):
		def sq(x):
			return x*x
## 		self.sys = GraphSystem(sq)
		self.sys = JayExample()
		self.s = SingleStepSolver(Spark(2), self.sys)
		self.s.initialize(u0=array([1.,1.,1.]), time=1)
## 		self.s.initialize(array([1.]))
	
	def test_run(self):
		self.s.run()
		print self.s.ts[-1]
		print self.s.us[-1]
		exact = self.sys.exact(self.s.ts[-1])
		print exact
		npt.assert_array_almost_equal(self.s.us[-1][:2], exact[:2], decimal=2)

class Test_Exceptions(object):
	limit = 20
	class LimitedSys(object):
		class LimitReached(Exception):
			pass
		def __init__(self, limit):
			self.limit = limit
			self.i = 0
		def f(self, t, x):
			if self.i < self.limit:
				self.i += 1
			 	return 0
			else:
				raise self.LimitReached()
	def setUp(self):
		self.sys = self.LimitedSys(self.limit)
		self.s = SingleStepSolver(ExplicitEuler(),self.sys)
		self.s.initialize(u0=0)
	
	def test_max_iter(self):
		self.max_iter = 1
		self.s.max_iter = self.max_iter
		try:
			self.s.run()
		except Solver.FinalTimeNotReached:
			npt.assert_equal(len(self.s.ts), self.max_iter + 1)
		else:
			raise Exception("FinalTimeNotReached not raised!")
	
	def test_sys_exception(self):
		try:
			self.s.run()
		except self.LimitedSys.LimitReached:
			npt.assert_equal(len(self.s.ts), self.limit + 1)
		else:
			raise Exception("Exception not raised")
	
def test_time():
	sys = System(lambda t,x: -x)
	sol = SingleStepSolver(ExplicitEuler(), sys)
	sol.h = Solver.time/10
	sol.initialize(u0=0.)
	sol.run(sol.h)
	npt.assert_(sol.ts[-1] < Solver.time)
	
		
import scipy.linalg as slin

class Test_LinearExponential(object):
	def test_run(self):
		for L in [np.array([[1.,2.],[3.,1.]]), -np.identity(2), ]: # np.zeros([2,2])
			for SchemeClass in [LawsonEuler, RKMK4T]:
				self.L = L
				print L
				self.sys = Linear(self.L)
				self.s = SingleStepSolver(RKMK4T(), self.sys)
				self.u0 = np.array([1.,0.])
				self.s.initialize(u0 = self.u0)
	
				h = self.s.scheme.h
				self.s.run(time=h)
				computed = self.s.us[-1]
				phi = Phi(0)
				tf = self.s.ts[-1]
				print tf
				phi_0 = np.dot(phi(tf*self.L)[0], self.u0)
				expected = np.dot(slin.expm(tf*self.L), self.u0)
				npt.assert_array_almost_equal(computed, expected)
				npt.assert_array_almost_equal(computed, phi_0)