# -*- coding: UTF-8 -*-
from __future__ import division

from odelab.scheme import *
from odelab.scheme.constrained import *

from odelab.system import *
from odelab.solver import *

import numpy.testing as npt

class Test_VerticalRollingDisk(object):
	def setUp(self):
		self.sys = VerticalRollingDisk()
		self.s = SingleStepSolver(McLachlan(), self.sys)
		ohm_phi = 2.
		ohm_theta = 1.
		phi_0 = 0
		self.u0 = array([0,0,phi_0,0.,0,0,ohm_phi,ohm_theta,0,0])
		R = self.sys.radius
		m = self.sys.mass
		vx = self.u0[4] = R*ohm_theta*np.cos(phi_0)
		vy = self.u0[5] = R*ohm_theta*np.sin(phi_0)
		self.s.initialize(self.u0)
		self.s.time = 1.
	
	def test_run(self):
		self.s.run(catch_runtime=False)
		npt.assert_array_almost_equal(self.s.final(), self.sys.exact(array([self.s.ts[-1]]),u0=self.u0)[:,0], decimal=1)
## 		self.s.plot(components=[8,9])

		
if __name__ == '__main__':
	from pylab import *
	t = Test_VerticalRollingDisk()
	t.setUp()
	t.test_run()
