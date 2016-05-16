#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import unittest


from odelab.scheme import *
from odelab.scheme.geometric import *

from odelab.system.mechanics import *
from odelab.solver import *

import numpy as np
import numpy.testing as npt


SingleStepSolver.catch_runtime = False

class Test_SymplecticEuler(unittest.TestCase):
	def _run(self):
		self.s = Solver(SymplecticEuler(h=.05), self.sys)
		self.s.initialize(u0=self.u0)
		self.s.run(5.)
		with self.s.open_store() as events:
			energy = self.s.system.energy(events)
			ef = energy[-1]
			e0 = energy[0]
		npt.assert_almost_equal(ef, e0, decimal=1)

	def test_harmosc(self):
		self.u0 = np.array([1.,0])
		self.sys = HarmonicOscillator()
		self._run()

	def test_henon(self):
		self.sys = HenonHeiles()
		H0 = 1/6
		self.u0 = self.sys.initial(H=H0,y=.1,py=.1)
		npt.assert_almost_equal(self.sys.energy(self.u0), H0)
		self._run()
