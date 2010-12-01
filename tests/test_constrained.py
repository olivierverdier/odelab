# -*- coding: UTF-8 -*-
from __future__ import division

from odelab.scheme import *
from odelab.scheme.constrained import *

from odelab.system import *
from odelab.solver import *

import numpy.testing as npt

import scipy.io

class Harness_VerticalRollingDisk(object):
	def setUp(self):
		self.sys = VerticalRollingDisk()
		self.setup_solver()
		ohm_phi = 2.
		ohm_theta = 1.
		phi_0 = 0
		self.u0 = array([0,0,phi_0,0.,0,0,ohm_phi,ohm_theta,0,0])
		R = self.sys.radius
		m = self.sys.mass
		# consistent initial velocities
		vx = self.u0[4] = R*ohm_theta*np.cos(phi_0)
		vy = self.u0[5] = R*ohm_theta*np.sin(phi_0)
		# lagrange multipliers: used only used as a guess in RK methods
		self.u0[8], self.u0[9] = -m*ohm_phi*R*ohm_theta*np.sin(phi_0), m*ohm_phi*R*ohm_theta*np.cos(phi_0)
		self.s.initialize(self.u0,h=.01)
		self.s.time = 1.
	
	def save(self, name):
		scipy.io.save_as_module('vrsave', {name: self.s})

	def test_run(self):
		self.s.run()
## 		self.s.plot(components=[6,7])
		npt.assert_array_almost_equal(self.s.final(), self.sys.exact(array([self.s.ts[-1]]),u0=self.u0)[:,0], decimal=1)

	def run(self, h, time):
		self.s.initialize(h=h,time=time)
		self.s.run()
## 			self.s.plot(components=[0,1,2,3,'qnorm'], error=True, save=run_name+'.pdf')

	experiences = {'50': (.01,50), '1000': (.1,1000.)}

	def save_exp(self, exp_name):
		h,time = self.experiences[exp_name]
		run_name = "%s_%s" % (type(self.s.scheme).__name__, exp_name)
		self.run(h, time)
		self.save(run_name)

	def run_1000(self):
		self.s.initialize(h=.1,time=1000)
		self.s.run()
		self.s.plot(components=[0,1,2,3,'qnorm'], error=True, save=type(self.s.scheme).__name__+'_1000.pdf')

class Test_VerticalRollingDisk_ML(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(McLachlan(), self.sys)

class Test_VerticalRollingDisk_H(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(NonHolonomicEnergy(), self.sys)

class Test_VerticalRollingDisk_Spark(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(Spark(2), self.sys)

		
if __name__ == '__main__':
	from pylab import *
	t = Test_VerticalRollingDisk_Spark()
	t.setUp()
	t.save_exp('50')
