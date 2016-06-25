
import unittest
import pytest
import numpy.testing as npt

import numpy as np

from odelab import Solver
from odelab.scheme.constrained import *
from odelab.scheme.rungekutta import *
from odelab.system.nonholonomic.rolling import VerticalRollingDisk, Robot

# Vertical Rolling Disk

class Harness_VerticalRollingDisk(object):
	h = .1
	def setUp(self):
		self.sys = VerticalRollingDisk()
		self.setup_scheme()
		ohm_phi = 2.
		ohm_theta = 1.
		phi_0 = 0
		self.u0 = self.sys.initial(np.array([0,0,phi_0,0.,0,0,ohm_phi,ohm_theta,0,0]))
		R = self.sys.radius
		m = self.sys.mass
		# consistent initial velocities
		vx = self.u0[4] = R*ohm_theta*np.cos(phi_0)
		vy = self.u0[5] = R*ohm_theta*np.sin(phi_0)
		# lagrange multipliers: used only used as a guess in RK methods
		self.u0[8], self.u0[9] = -m*ohm_phi*R*ohm_theta*np.sin(phi_0), m*ohm_phi*R*ohm_theta*np.cos(phi_0)
		self.scheme.h = self.h
		self.s = Solver(self.scheme, self.sys)
		self.s.initialize(self.u0,)
		self.s.time = 1.

	def test_run(self):
		self.s.run()
## 		self.s.plot(components=[6,7])
		self.check_solution()

	def check_solution(self, decimal=1):
		npt.assert_array_almost_equal(self.s.final()[:8], self.sys.exact(np.array([self.s.final_time()]),u0=self.u0)[:8,0], decimal=decimal)

	def check_energy(self, decimal):
		energy = self.sys.energy
		npt.assert_almost_equal(energy(self.s.final()), energy(self.s.initial()), decimal=decimal)

	experiences = {'50': (.01,50), '1000': (.1,1000.)}

	def run_experience(self, exp_name):
		h,time = self.experiences[exp_name]
		self.s.initialize(h=h,time=time)
		self.s.run()


class Test_VerticalRollingDisk_ML(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = McLachlan()

class Test_VerticalRollingDisk_H(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = NonHolonomicEnergy()

class Test_VerticalRollingDisk_H0(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = NonHolonomicEnergy0()

class Test_VerticalRollingDisk_HM(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = NonHolonomicEnergyEMP()

class Test_VerticalRollingDisk_Spark2(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = Spark2()

class Test_VerticalRollingDisk_Spark3(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = Spark3()

class Test_VerticalRollingDisk_Spark4(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = Spark4()

class Test_VerticalRollingDisk_SE(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = SymplecticEuler()

class Test_VerticalRollingDisk_LF(Harness_VerticalRollingDisk, unittest.TestCase):
	def setup_scheme(self):
		self.scheme = NonHolonomicLeapFrog()
