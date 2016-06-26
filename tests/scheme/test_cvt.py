# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import pytest
import numpy.testing as npt

from odelab.scheme import *
from odelab.scheme.constrained import *

from odelab.system import *
from odelab.system.nonholonomic.contactoscillator import *
from odelab.system.nonholonomic.pendulum import Pendulum, CirclePendulum, SinePendulum
from odelab.system.nonholonomic.rolling import VerticalRollingDisk, Robot
from odelab.solver import *

from odelab.scheme.rungekutta import *


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




@pytest.mark.parametrize(['scheme', 'energy_tol'], [(NonHolonomicEnergy(), 2), (McLachlan(), 2)], ids=repr)
def test_energy(scheme, energy_tol):
	scheme.h = .05
	s = SingleStepSolver(scheme, ChaoticOscillator(3))
	u0 = np.zeros(15)
	N = 10 - 1
	n = 2 # j in range(N+1)
	angle = n*np.pi/2/N
	u0[:7] = np.array([np.cos(angle),.6,.4,.2,1.,1.,1.])
	u0[7:9] = np.array([0., np.sin(angle)])
	s.initialize(u0=u0,  time=1)

	s.run()
	H1 = s.system.energy(s.final())
	H0 = s.system.energy(s.initial())
	npt.assert_almost_equal(H1, H0, decimal=energy_tol)


# Old test below

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

