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

class TestOsc(object):
	epsilon = 0.3
	def setUp(self, scheme, system):
		self.scheme = scheme
		self.system = system
		# self.sys = ContactOscillator(epsilon=self.epsilon)
		self.s = Solver(scheme, system)
		#self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
		self.s.time = 10.



	z0s = np.linspace(-.9,.9,10)*np.sqrt(2)
	N = 15

	@pytest.mark.parametrize(['scheme', 'system', 'decimal'], [
		(McLachlan(), ContactOscillator(epsilon=.3), 1),
		(Spark2(), ContactOscillator(epsilon=.3), 1),
		(Spark3(), ContactOscillator(epsilon=.3), 1),
		(NonHolonomicEnergy(), ContactOscillator(epsilon=.3), 6),
		(McLachlan(), NonReversibleContactOscillator(), 1),
		(NonHolonomicEnergy(), NonReversibleContactOscillator(), 10),
		(Spark2(), NonReversibleContactOscillator(), 1),
		(Spark3(), NonReversibleContactOscillator(), 1),
	],
							 ids=repr)
	def test_z0(self, scheme, system, decimal, i=5, nb_Poincare_iterations=1):
		self.setUp(scheme, system)
		z0 = self.z0s[i]
		h = system.time_step(self.N)
		scheme.h = h
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=system.initial_sin(z0,), time=time)
		self.s.run()
		#self.s.plot(['radius'])
		npt.assert_almost_equal(system.energy(self.s.final()), system.energy(self.s.initial()), decimal=decimal)
		with self.s.open_store() as events:
			energy = self.s.system.energy(events)

	def run_chaotic(self, nb_Poincare_iterations=10):
		z0 = -.55
		h = self.sys.time_step(self.N)
		self.scheme.h = h
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial_cos(z0), h=h, time=time)
		self.s.run()


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

