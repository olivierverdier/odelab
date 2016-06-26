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


