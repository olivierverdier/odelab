#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.solver import SingleStepSolver
from odelab.system import System
from odelab.experiment import Experiment

import numpy as np
import nose.tools as nt

def f(t,u):
	return -u

class Test_Experiment(object):
	def setUp(self):
		from odelab.scheme import ExplicitEuler
		params = {
			'family': 'tmpbank',
			'name': 'tmpexp',
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
		exp = Experiment(params)
		exp.run()

	def test_load(self):
		s = Experiment.load('tmpbank', 'tmpexp')
		nt.assert_true(isinstance(s, SingleStepSolver))
		nt.assert_equal(s.scheme.__class__.__name__, 'ExplicitEuler')
		nt.assert_equal(len(s), 11)

