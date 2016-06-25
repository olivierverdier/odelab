# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import pytest


from odelab.scheme import *
from odelab.scheme.constrained import *

from odelab.system import *
from odelab.system.nonholonomic.contactoscillator import *
from odelab.system.nonholonomic.pendulum import Pendulum, CirclePendulum, SinePendulum
from odelab.system.nonholonomic.rolling import VerticalRollingDisk, Robot
from odelab.solver import *

from odelab.scheme.rungekutta import *

import numpy.testing as npt

SingleStepSolver.catch_runtime = False


# Contact oscillator

class Harness_Osc(object):
	epsilon = 0.3
	def setUp(self):
		self.sys = ContactOscillator(epsilon=self.epsilon)
		self.set_scheme()
		self.s = SingleStepSolver(self.scheme, self.sys)
		#self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
		self.s.time = 10.



	z0s = np.linspace(-.9,.9,10)*np.sqrt(2)
	N = 15

	decimal = 1

	def test_z0(self, i=5, nb_Poincare_iterations=1):
		z0 = self.z0s[i]
		h = self.sys.time_step(self.N)
		self.scheme.h = h
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial_sin(z0,), time=time)
		self.s.run()
		#self.s.plot(['radius'])
		npt.assert_almost_equal(self.sys.energy(self.s.final()), self.sys.energy(self.s.initial()), decimal=self.decimal)
		with self.s.open_store() as events:
			energy = self.s.system.energy(events)

	def run_chaotic(self, nb_Poincare_iterations=10):
		z0 = -.55
		h = self.sys.time_step(self.N)
		self.scheme.h = h
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial_cos(z0), h=h, time=time)
		self.s.run()

class Test_Initial(Harness_Osc, unittest.TestCase):
	def set_scheme(self):
		self.scheme = McLachlan()

	def test_initial(self):
		u0 = self.sys.initial_cos(self.z0s[5])
		u00 = ContactSystem.initial(self.sys, u0)
		self.assertFalse(u0 is u00)
		npt.assert_almost_equal(u0,u00)


class Test_McOsc(Harness_Osc, unittest.TestCase):
	label = 'ML'
	def set_scheme(self):
		self.scheme = McLachlan()

class Test_JayOsc2(Harness_Osc, unittest.TestCase):
	def set_scheme(self):
		self.scheme = Spark2()

class Test_JayOsc3(Harness_Osc, unittest.TestCase):
	def set_scheme(self):
		self.scheme = Spark3()

class Test_HOsc(Harness_Osc, unittest.TestCase):
	decimal = 6
	label = 'H'
	def set_scheme(self):
		self.scheme = NonHolonomicEnergy()

class Test_NROsc(Test_McOsc):
	epsilon = 0.
	def setUp(self):
		self.sys = NonReversibleContactOscillator()
		self.set_scheme()
		self.s = SingleStepSolver(self.scheme, self.sys)
		self.s.time = 10.

class Test_NROsc_H(Test_NROsc):
	decimal = 10
	def set_scheme(self):
		self.scheme = NonHolonomicEnergy()

class Test_NROsc_SP2(Test_NROsc):
	def set_scheme(self):
		self.scheme = Spark2()

class Test_NROsc_SP3(Test_NROsc):
	def set_scheme(self):
		self.scheme = Spark3()

class Test_OscSolver(unittest.TestCase):
	def get_ht(self, z0,N,P):
		u"""
		N: nb iteration per Poincaré iteration
		P: nb of Poincaré iterations
		"""
		h = 2*np.sin(np.pi/N) # is this correct?
		time = P*N*h
		return h,time

	def setUp(self, N=40, P=100,):
		z0 = -.5*np.sqrt(2)
		print('z0 =',z0)
		sys = ContactOscillator(epsilon=0)
		u0 = sys.initial_cos(z0)
		h,time = self.get_ht(z0, N,P)
		solver = SingleStepSolver(McLachlan(h), sys,)
		solver.initialize(time=time, u0=u0, )
		self.solver = solver

	def notest_run(self):
		self.solver.run()



class TestRobot(object):
	def setUp(self, scheme):
		self.scheme = scheme
		s = SingleStepSolver(self.scheme, Robot())
		u0 = np.zeros(10)
		u0[4] = 1.
		u0[7] = 1.
		self.scheme.h = .2
		s.initialize(time=1, u0 = u0)
		self.s = s

	@pytest.mark.parametrize('scheme', [NonHolonomicEnergy(), McLachlan()], ids=repr)
	def test_run(self, scheme):
		self.setUp(scheme)
		self.s.run()

	@pytest.mark.parametrize('scheme', [NonHolonomicEnergy(), NonHolonomicEnergy0(), NonHolonomicEnergyEMP()], ids=repr)
	def test_energy(self, scheme):
		self.setUp(scheme)
		s = self.s
		s.run(time=10)
		npt.assert_almost_equal(s.system.energy(s.final()), s.system.energy(s.initial()), decimal=4)


# Test Spark on simple ODE

def minus_time(tx):
	return -tx[0]

class Test_SparkODE(unittest.TestCase):
	def setUp(self):
		self.sys = ODESystem(minus_time)
		scheme = Spark4()
		scheme.h = .1
		self.s = SingleStepSolver(scheme, self.sys)
		self.s.initialize(array([1.]))

	def test_run(self):
		self.s.run()
		exact = np.exp(-self.s.get_times())
		print(exact[-1])
		print(self.s.final())
## 		npt.assert_array_almost_equal(self.s.aus, exact, 5)
		npt.assert_almost_equal(self.s.final()[0], exact[-1])
## 		plot(self.s.ats, np.vstack([self.s.aus, exact]).T)

# Test spark with Jay Example as a system

class Test_JayExample(unittest.TestCase):
	def setUp(self):
		self.sys = JayExample()
## 		self.s.initialize(array([1.]))

	def test_spark(self):
		scheme = Spark2()
		scheme.h = .05
		self.s = SingleStepSolver(scheme, self.sys)
		self.s.initialize(u0=array([1.,1.,1.]), time=1,)
		self.s.run()
		print(self.s.final_time())
		print(self.s.final())
		exact = self.sys.exact(self.s.final_time(),array([1.,1.,1.]))
		print(exact)
		npt.assert_array_almost_equal(self.s.final()[:2], exact[:2], decimal=2)

def test_pendulum_ML():
	s = SingleStepSolver(McLachlan(h=.1), CirclePendulum())
	s.initialize(np.array([1.,0,0,0,0]))
	s.run()

def test_pendulum_NHE():
	s = SingleStepSolver(NonHolonomicEnergy(h=.1), SinePendulum())
	s.initialize(np.array([1.,0,0,0,0]))
	s.run()

class Harness_chaoticosc(object):
	def setUp(self):
		self.scheme.h = .05
		s = SingleStepSolver(self.scheme, ChaoticOscillator(3))
		u0 = np.zeros(15)
		N = 10 - 1
		n = 2 # j in range(N+1)
		angle = n*np.pi/2/N
		u0[:7] = np.array([np.cos(angle),.6,.4,.2,1.,1.,1.])
		u0[7:9] = np.array([0., np.sin(angle)])
		s.initialize(u0=u0,  time=1)
		self.s = s

	def test_run(self):
		s = self.s
		s.run()
		print(s.system.energy(s.final()))

	def test_energy(self):
		self.s.run()
		H1 = self.s.system.energy(self.s.final())
		H0 = self.s.system.energy(self.s.initial())
		print(H0)
		print(H1)
		npt.assert_almost_equal(H1, H0, decimal=self.energy_tol)

class Test_chaotic_ML(Harness_chaoticosc, unittest.TestCase):
	scheme = McLachlan()
	energy_tol = 2

class Test_chaotic_H(Harness_chaoticosc, unittest.TestCase):
	scheme = NonHolonomicEnergy()
	energy_tol = 10

@pytest.mark.parametrize('scheme', [McLachlan(), NonHolonomicEnergy()], ids=repr)
def test_chaplygin(scheme):
	solver = SingleStepSolver(scheme, Chaplygin(g=.1))
	#u0 = np.array([1.,0,.2,0,0,0,0])
	#u0 = np.array([1.,0,.8*np.pi/2,0,0,0,0])
	u0_Hilsen = np.array([1.,0,0,0,0,0,0])
	h_Hilsen = 1./300
	time_Hilsen = 1.
	u0_Jay = np.array([0,0,0,0,0,1.,0])
	h_Jay = .1
	time_Jay = 100
	scheme.h = h_Jay
	#self.s.initialize(u0=u0_Jay,time=time_Jay,)
	solver.initialize(u0=u0_Jay,time=1)
	#print self.s.system.energy(s.final())
	#nt.assert_equal(s.system.energy(self.s.events_array).shape, (len(self.s),))
	#return self.s

	solver.run()
	#nt.assert_almost_equal(s.system.energy(s.initial()), s.system.energy(s.final()))


