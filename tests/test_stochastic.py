# -*- coding: UTF-8 -*-
from __future__ import division

import odelab

from odelab.scheme.stochastic import *
from odelab.system import *
from odelab.solver import *

import pylab as pl

import numpy as np

class Test_OU(object):
	def test_run(self):
		sys = OrnsteinUhlenbeck()
		self.s = SingleStepSolver(EulerMaruyama(), sys)
		self.s.initialize(u0=np.array([1.]), h=.01, time=1.)
		self.s.run()

class Test_Differentiator(object):
	t0 = 5e-9
 	V0 = .01
	def test_run(self):
		sys = Differentiator(LinBumpSignal(self.V0,self.t0))
## 		sys.kT = 0. # no noise
		self.s = SingleStepSolver(EulerMaruyama(), sys)
		self.s.initialize(u0 = np.array([0,0,0,0,0.]), h=2.5e-11, time=5*self.t0)
		self.s.run()
		

import sys
sys.path.append('/Users/olivierpb/Documents/latex/geode/kronecker')

if __name__ == '__main__':
	t = Test_Differentiator()
	t.test_run()
	t.s.plot(2)
