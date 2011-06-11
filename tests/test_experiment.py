#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.solver import SingleStepSolver
from odelab.system import System
from odelab.experiment import Experiment

import numpy as np
import nose.tools as nt

import tempfile
import os

def f(t,u):
	return -u

class Test_Experiment(object):
	name = 'tmpexp'
	def setUp(self):
		self.file = tempfile.NamedTemporaryFile()
		self.path = self.file.name
		self.family = os.path.basename(self.path)
		self.prefix = os.path.dirname(self.path)
		self.file.close()
		from odelab.scheme import ExplicitEuler
		params = {
			'family': self.family,
			'name': self.name,
			'system': System,
			'system_params': {'f': f},
			'solver': SingleStepSolver,
			'scheme': ExplicitEuler,
			'scheme_params': {},
			'initialize': {
				'u0' : np.array([1.]),
				'time': 1.,
				'h': .1,
				},
			}
		exp = Experiment(params, store_prefix=self.prefix)
		exp.run()
		exp.solver.file.close()

	def test_load(self):
		s = Experiment.load(self.path, self.name)
		nt.assert_true(isinstance(s, SingleStepSolver))
		nt.assert_equal(s.scheme.__class__.__name__, 'ExplicitEuler')
		nt.assert_equal(len(s), 11)
	
	#def tearDown(self):
		#import os
		#os.remove()

if __name__ == '__main__':
	t = Test_Experiment()
	t.setUp()
