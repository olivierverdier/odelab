from odelab.solver import *
from odelab.system import *

import numpy.testing as npt


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
		self.s.initialize(u0=self.sys.initial(z0), h=self.sys.time_step(self.N))
		self.s.time = nb_Poincare_iterations*self.N*self.s.h
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
		self.s = McLachlan(self.sys)

class Test_JayOsc(Harness_Osc):
	def set_solver(self):
		self.s = Spark(self.sys, 2)
	
class Test_SparkODE(object):
	def setUp(self):
		def f(xt):
			return -xt[0]
		self.sys = ODESystem(f)
		self.s = Spark(self.sys, 4)
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
		self.s = Spark(self.sys, 2)
		self.s.initialize(u0=array([1.,1.,1.]), time=1)
## 		self.s.initialize(array([1.]))
	
	def test_run(self):
		self.s.run()
		print self.s.ts[-1]
		print self.s.us[-1]
		exact = self.sys.exact(self.s.ts[-1])
		print exact
		npt.assert_array_almost_equal(self.s.us[-1][:2], exact[:2], decimal=2)

class Test_Simple(object):
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
		self.s = ExplicitEuler(self.sys)
		self.s.initialize(u0=0)
	
	def test_max_iter(self):
		self.max_iter = 1
		self.s.max_iter = self.max_iter
		try:
			self.s.run()
		except ODESolver.FinalTimeNotReached:
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
	sol = ExplicitEuler(sys)
	sol.h = ExplicitEuler.time/10
	sol.initialize(u0=0.)
	sol.run(sol.h)
	npt.assert_(sol.ts[-1] < ExplicitEuler.time)
	
		
import scipy.linalg as slin

class Test_LinearExponential(object):
	def test_run(self):
		for L in [np.array([[1.,2.],[3.,1.]]), -np.identity(2), ]: # np.zeros([2,2])
			self.L = L
			print L
			self.sys = Linear(self.L)
			self.s = LawsonEuler(self.sys)
			self.u0 = np.array([1.,0.])
			self.s.initialize(u0 = self.u0)
	
			h = self.s.h
			self.s.run(time=h)
			computed = self.s.us[-1]
			phi = Phi(0)
			tf = self.s.ts[-1]
			print tf
			phi_0 = np.dot(phi(tf*self.L)[0], self.u0)
			expected = np.dot(slin.expm(tf*self.L), self.u0)
			npt.assert_array_almost_equal(computed, expected)
			npt.assert_array_almost_equal(computed, phi_0)