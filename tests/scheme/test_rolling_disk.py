
import unittest
import pytest
import numpy.testing as npt

import numpy as np

from odelab import Solver
from odelab.scheme.constrained import *
from odelab.scheme.rungekutta import *
from odelab.system.nonholonomic.rolling import VerticalRollingDisk, Robot

schemes = [
	McLachlan(),
	NonHolonomicEnergy(),
	NonHolonomicEnergy0(),
	NonHolonomicEnergyEMP(),
	Spark2(),
	Spark3(),
	Spark4(),
	SymplecticEuler(),
	NonHolonomicLeapFrog(),
]

@pytest.fixture(params=schemes, ids=repr)
def scheme(request):
	return request.param


class TestVerticalRollingDisk:
	h = .1
	def test_exact(self, scheme):
		self.sys = VerticalRollingDisk()
		self.scheme = scheme
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
		self.s.run(1.)
##		self.s.plot(components=[6,7])
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

