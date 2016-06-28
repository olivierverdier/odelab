# -*- coding: utf-8 -*-
from __future__ import division

import unittest

import odelab

from odelab.scheme.stochastic import *
from odelab.system import *
from odelab.solver import *


import numpy as np

class Test_OU(unittest.TestCase):
	def test_run(self):
		sys = OrnsteinUhlenbeck()
		scheme = EulerMaruyama()
		scheme.h = .01
		self.s = SingleStepSolver(scheme, sys)
		self.s.initialize(u0=np.array([1.]))
		self.s.run(time=1.)

class Test_Differentiator(unittest.TestCase):
	t0 = 5e-9
	V0 = .01
	def test_run(self):
		sys = Differentiator(LinBumpSignal(self.V0,self.t0))
## 		sys.kT = 0. # no noise
		scheme = EulerMaruyama()
		## scheme.h = 2.5e-11
		scheme.h = self.t0
		self.s = SingleStepSolver(scheme, sys)
		self.s.initialize(u0 = np.array([0,0,0,0,0.]))
		self.s.run(time=5*self.t0)


