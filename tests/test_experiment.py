#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.solver import SingleStepSolver, MultiStepSolver, load_solver
from odelab.system import System
from odelab.experiment import Experiment

import numpy as np
import nose.tools as nt
from nose.plugins.skip import SkipTest
import numpy.testing as npt

import tempfile
import os

def f(t,u):
	return -u

class BigSystem(System):
	def __init__(self, *args, **kwargs):
		super(BigSystem, self).__init__(*args, **kwargs)
		self.data = np.zeros([32,32], dtype=complex)

class Harness_Experiment(object):
	name = 'tmpexp'
	def setUp(self):
		self.file = tempfile.NamedTemporaryFile()
		self.path = self.file.name
		self.family = os.path.basename(self.path)
		self.prefix = os.path.dirname(self.path)
		self.file.close()
		from odelab.scheme import ExplicitEuler
		s = self.solver_class(system=System(f), scheme=ExplicitEuler(h=.1), path=self.path)
		s.catch_runtime = False

		params = {
			'family': self.family,
			'system': System,
			'system_params': {'f': f},
			'solver': SingleStepSolver,
			'scheme': ExplicitEuler,
			'scheme_params': {},
			'initialize': {
				'u0' : np.array([1.]),
				'time': 1.,
				'name': self.name,
				},
			}
		s.initialize(**params['initialize'])
		s.run()
		#exp = Experiment(params, store_prefix=self.prefix)
		#exp.run()
		#exp.solver.file.close()

	def test_load(self):
		s = load_solver(self.path, self.name)
		#s = Experiment.load(self.path, self.name)
		nt.assert_true(isinstance(s, SingleStepSolver))
		nt.assert_equal(s.scheme.__class__.__name__, 'ExplicitEuler')
		nt.assert_equal(len(s), 11)
		with s.open_store() as events:
			nt.assert_equal(len(events), 11)
			npt.assert_array_almost_equal(events[-1], np.linspace(0,1,11))

	def test_load_run(self):
		raise SkipTest('Not possible to restart a stored simulation')
		s = load_solver(self.path, self.name)
		s.run()

class Test_Single(Harness_Experiment):
	solver_class = SingleStepSolver

class Test_Multi(Harness_Experiment):
	def setUp(self):
		raise SkipTest("Multistep solvers do not work")
	solver_class = MultiStepSolver
