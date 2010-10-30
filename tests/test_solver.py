# -*- coding: UTF-8 -*-
from __future__ import division


from odelab.scheme import *
from odelab.scheme.constrained import *
from odelab.scheme.exponential import *

from odelab.system import *
from odelab.solver import *

import tempfile
import os

import numpy.testing as npt
import nose.tools as nt

import pylab as pl
pl.ioff()

class Harness(object):
	no_plot = True

def f(t,u):
	return t*np.ones_like(u)

def const_f(c,t,u):
	return c*np.ones_like(u)

def time_f(t,u):
	return t

from functools import partial
const_r = partial(const_f, 1.)
const_c = partial(const_f, 1.j)

class Harness_Solver(Harness):
	dim = 1
	def set_system(self, f):
		self.solver.system = System(f)
	
	def test_scheme_str(self):
		# should not raise an exception even though h is not yet set in the underlying scheme:
		print str(self.solver)
	
	def test_initialize(self):
		u0 = np.random.rand(self.dim)
		self.solver.initialize(u0=u0)
		nt.assert_true(self.solver.time == Solver.time)
		nt.assert_true(len(self.solver.ts) == 1)
		nt.assert_true(len(self.solver.us) == 1)
	
	def test_initialize_scheme(self):
		h = 10.
		self.solver.initialize(u0=np.random.rand(self.dim),h=h)
		self.solver.step(self.solver.ts[0], self.solver.initial())
		nt.assert_true(self.solver.scheme.h == h)
	
	def test_quadratic(self):
		"""should solve f(t) = t pretty well"""
		print type(self).__name__
		self.set_system(time_f)
		self.solver.initialize(u0=1., time=1.)
		self.solver.run()
		# u'(t) = t; u(0) = u0; => u(t) == u0 + t**2/2
		nt.assert_almost_equal(self.solver.final(), 3/2, 1)

	def check_const(self, f, u0, expected):
		"""should solve the f=c exactly"""
		print type(self).__name__
		self.set_system(f)
		self.solver.initialize(u0=1.+0j, time=1.)
		self.solver.run()
		npt.assert_almost_equal(self.solver.final(), expected, 1)

	def test_const(self):
		for f,u0,expected in [(const_r, 1., 2.), (const_c, 1.+0j, 1.+1.j), (const_c, 1., 1.+1.j)]:
			yield self.check_const, f, u0, expected

class Test_EEuler(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(ExplicitEuler(), System(f))

class Test_RK4(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(RungeKutta4(), System(f))

class Test_RK34(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(RungeKutta34(), System(f))

class Test_ode15s(Harness_Solver):
	def setUp(self):
		self.solver = SingleStepSolver(ode15s(), System(f))

class Test_LawsonEuler(Harness_Solver):
	def set_system(self, f):
		self.solver.system = NoLinear(f,self.dim)
	def setUp(self):
		self.solver = SingleStepSolver(LawsonEuler(), NoLinear(f,self.dim))

## class Test_IEuler(Harness_Solver):
## 	def setUp(self):
## 		self.solver = SingleStepSolver(ImplicitEuler, System(f))

@nt.raises(Solver.Unstable)
def test_unstable():
	s = SingleStepSolver(LawsonEuler(), Linear(np.array([[1.e2]])))
	s.initialize(u0 = 1., time = 100, h = 10)
	s.run()

	
class Harness_Circle(Harness):
	def setUp(self):
		def f(t,u):
			return array([-u[1], u[0]])
		self.f = f
		self.make_solver()
		self.s.initialize(u0 = array([1.,0.]), h=.01, time = 10.)
	
	def test_plot(self):
		self.s.run()
		self.s.plot2D()
		self.s.plot(plot_exact=False)
		self.s.plot(plot_exact=True)
		tmp = tempfile.gettempdir()
		path = os.path.join(tmp, 'test_fig.pdf')
		print path
		self.s.plot(save=path)

class Test_Circle_EEuler(Harness_Circle):
	def make_solver(self):
		self.s = SingleStepSolver(ExplicitEuler(), System(self.f))

class Test_Circle_RK34(Harness_Circle):
	def make_solver(self):
		self.s = SingleStepSolver(RungeKutta34(), System(self.f))

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
		h = self.sys.time_step(self.N)
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial(z0), h=h, time=time)
		self.s.run()
		npt.assert_almost_equal(self.sys.energy(self.s.final()), self.sys.energy(self.s.initial()), decimal=1)

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
	N=5 # bigger time step to make Jay test faster
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
		print self.s.final()
## 		npt.assert_array_almost_equal(self.s.aus, exact, 5)
		npt.assert_almost_equal(self.s.final(), exact[-1])
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
		print self.s.final()
		exact = self.sys.exact(self.s.ts[-1],array([1.,1.,1.]))
		print exact
		npt.assert_array_almost_equal(self.s.final()[:2], exact[:2], decimal=2)

class DummyException(Exception):
	pass

class Test_FinalTimeExceptions(object):
	limit = 20
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
	
	@nt.raises(Solver.Runtime)
	def test_sys_exception(self):
		self.s.run()

	@nt.raises(DummyException)
	def test_sys_no_runtime_exception(self):
		self.s.run(catch_runtime=False)

class Test_Exceptions(object):
	def setUp(self):
		self.s = SingleStepSolver(ExplicitEuler(), Linear(np.array([[1]])))
	@nt.raises(Solver.NotInitialized)
	def test_no_u0(self):
		self.s.initialize()
	@nt.raises(Solver.NotInitialized)
	def test_no_initialize(self):
		self.s.run()
	@nt.raises(Solver.Unstable)
	def test_unstable(self):
		self.s = SingleStepSolver(ExplicitEuler(), Linear(np.array([[float('inf')]])))
		self.s.initialize(u0=np.array([0]))
		self.s.run()
	@nt.raises(Solver.Runtime)
	def test_runtime_exception(self):
		def f(t,u):
			raise Exception('message')
		self.s = SingleStepSolver(ExplicitEuler(), System(f))
		self.s.initialize(u0=0)
		self.s.run()

class Test_Simple(object):
	def setUp(self):
		class TotSys(System):
			def total(self, xt):
				return np.sum(xt[:-1],axis=0)
		sys = TotSys(lambda t,x: -x)
		self.s = SingleStepSolver(ExplicitEuler(), sys)
	
	def test_time(self):
		sol = self.s
		sol.h = Solver.time/10
		sol.initialize(u0=0.)
		sol.run(sol.h)
		npt.assert_(sol.ts[-1] < Solver.time)

	def test_extra_run(self):
		"""test that an extra run continues from last time"""
		sol = self.s
		sol.initialize(u0=1.)
		sol.run()
		npt.assert_almost_equal(sol.ts[-1],Solver.time)
		sol.run()
		npt.assert_almost_equal(sol.ts[-1],2*Solver.time)
	
	def test_plot_args(self):
		self.s.initialize(u0=np.array([1.,1.,1.]))
		self.s.run()
		lines = self.s.plot(0,lw=5)
		npt.assert_equal(len(lines),1)
		lines = self.s.plot(lw=5)
		npt.assert_equal(len(lines),3)
		npt.assert_equal(lines[-1].get_linewidth(),5)
	
	def test_plot_function(self):
		self.s.initialize(u0=np.array([1.,1.,1.]))
		self.s.run()
		lines = self.s.plot_function('total', lw=4)
		npt.assert_equal(lines[-1].get_linewidth(), 4)
		


import scipy.linalg as slin

class CompareLinearExponential(object):
	def __init__(self, scheme):
		self.description = 'Compare exponential for %s' % type(scheme).__name__
	def __call__(self, computed, expected, phi):
		npt.assert_array_almost_equal(computed, expected)
		npt.assert_array_almost_equal(computed, phi)

def test_linear_exponential():
	for L in [np.array([[1.,2.],[3.,1.]]), -np.identity(2), ]: # np.zeros([2,2])
		for scheme in [
			LawsonEuler(), 
			RKMK4T(), 
			HochOst4(), 
			ABLawson2(), 
			ABLawson3(), 
			ABLawson4(),
			Lawson4(),
			ABNorset4(),
			GenLawson45(),
		]:
			sys = Linear(L)
			s = MultiStepSolver(scheme, sys)
			u0 = np.array([1.,0.])
			h = .1
			s.initialize(u0 = u0, h=h)

			s.run(time=1.)
			computed = s.final()
			phi = Phi(0)
			tf = s.ts[-1]
			phi_0 = np.dot(phi(tf*L)[0], u0)
			expected = np.dot(slin.expm(tf*L), u0)
			yield CompareLinearExponential(scheme), computed, expected, phi_0


import pylab as pl

class Harness_ComplexConvection(object):

	def check_convection(self, do_plot):
		scheme = self.scheme
		h = self.time/self.N
		self.s = MultiStepSolver(scheme, self.B)
		self.s.initialize(u0=self.u0, time=self.time, h=h)
		print scheme
		self.s.run()
		u1 = self.s.final()
		if np.any(np.isnan(u1)):
			raise Exception('unstable!')
		if do_plot:
			pl.plot(self.B.points, self.u0)
			pl.plot(self.B.points, u1)
		npt.assert_array_almost_equal(u1, self.sol, decimal=2)

	def test_run(self, do_plot=False):
		self.B = BurgersComplex(viscosity=0., size=32)
		umax=.5
		self.u0 = 2*umax*(.5 - np.abs(self.B.points))
		self.time = .5
		mid = self.time*umax # the peak at final time
		self.sol = (self.B.points+.5)*umax/(mid+.5)*(self.B.points < mid) + (self.B.points-.5)*umax/(mid-.5)*(self.B.points > mid)
		if do_plot:
			pl.clf()
			pl.plot(self.B.points, self.sol, lw=2)
		self.check_convection(do_plot)
	
	def find_N(self):
		for N in [10,20,50,75,100,120,150]:
			self.N = N
			try:
				self.notest_run()
			except AssertionError:
				continue
			else:
				break
		else:
			raise Exception('No N!')
		print type(self.scheme).__name__, N
		
class Test_CC_EE(Harness_ComplexConvection):
	scheme = ExplicitEuler()
	N=150

class Test_CC_RK4(Harness_ComplexConvection):
	scheme = Kutta(4)
	N = 10

class Test_CC_RK38(Harness_ComplexConvection):
	scheme = Kutta(38)
	N = 10

class Test_CC_Heun(Harness_ComplexConvection):
	scheme = Heun()
	N = 10

class Test_CC_ABN4(Harness_ComplexConvection):
	scheme = ABNorset4()
	N = 50

class Test_CC_ABL2(Harness_ComplexConvection):
	scheme = ABLawson2()
	N = 50

class Test_CC_ABL3(Harness_ComplexConvection):
	scheme = ABLawson3()
	N=50

class Test_CC_ABL4(Harness_ComplexConvection):
	scheme = ABLawson4()
	N=50

class Test_CC_L4(Harness_ComplexConvection):
	scheme = Lawson4()
	N=10

class Test_CC_GL45(Harness_ComplexConvection):
	scheme = GenLawson45()
	N=10

class Test_CC_LE(Harness_ComplexConvection):
	scheme = LawsonEuler()
	N=150

class Test_CC_RKMK4T(Harness_ComplexConvection):
	scheme = RKMK4T()
	N=10

class Test_CC_ode15s(Harness_ComplexConvection):
	scheme = ode15s()
	N=2

class Test_CC_AB1(Harness_ComplexConvection):
	scheme = AdamsBashforth(1)
	N = 150

class Test_CC_AB2(Harness_ComplexConvection):
	scheme = AdamsBashforth(2)
	N = 50

class Test_CC_AB2e(Harness_ComplexConvection):
	scheme = AdamsBashforth('2e')
	N = 50

class Test_CC_B1(Harness_ComplexConvection):
	scheme = Butcher(1)
	N=50

class Test_CC_B3(Harness_ComplexConvection):
	scheme = Butcher(3)
	N=120


class CheckSquare(object):
	def __init__(self, name):
		self.description = name
	def __call__(self,name,a,b,nb_stages, tail_length):
		npt.assert_equal(len(b), tail_length, 'bv not right # of rows')
		for i,row in enumerate(a):
			npt.assert_equal(len(row), nb_stages+tail_length+1, 'au[%d]'%i)
		for i,row in enumerate(b):
			npt.assert_equal(len(row), nb_stages+tail_length, 'bv[%d]'%i)

def test_exp_square():
	"""
	Check that the matrices produced by the exponential schemes are square.
	"""
	import odelab.scheme.exponential as E
	for name in dir(E):
		cls = getattr(E, name)
		if hasattr(cls, 'general_linear_z'):
			obj = cls()
			a,b = obj.general_linear_z(np.eye(2))
			nb_stages = len(a)
			tail_length = obj.tail_length
			yield CheckSquare(name),name, a,b, nb_stages, tail_length

