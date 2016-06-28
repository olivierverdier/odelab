#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import unittest
import pytest

from odelab.solver import Solver, load_solver
from odelab.system import System
from odelab.experiment import Experiment

import numpy as np
import numpy.testing as npt

import tempfile
import os

from odelab.store import Store, SimpleStore
pytestmark = pytest.mark.skipif(Store is SimpleStore, reason="Store tests only make sense with pytables")

def f(t,u):
	return -u

class BigSystem(System):
	def __init__(self, *args, **kwargs):
		super(BigSystem, self).__init__(*args, **kwargs)
		self.data = np.zeros([32,32], dtype=complex)

class Test_Experiment(unittest.TestCase):
	name = 'tmpexp'
	def setUp(self):
		self.file = tempfile.NamedTemporaryFile()
		self.path = self.file.name
		self.family = os.path.basename(self.path)
		self.prefix = os.path.dirname(self.path)
		self.file.close()
		from odelab.scheme.classic import ExplicitEuler

		self.params = params = {
			'family': self.family,
			'system': System,
			'system_params': {'f': f},
			'solver': Solver,
			'scheme': ExplicitEuler,
			'scheme_params': {
				'h': .1,
				},
			'initialize': {
				'u0' : np.array([1.]),
				'name': self.name,
				},
			}
		scheme = ExplicitEuler(**params['scheme_params'])
		s = Solver(system=System(f), scheme=scheme, path=self.path)
		s.catch_runtime = False
		s.initialize(**params['initialize'])
		s.run(1.)
		#exp = Experiment(params, store_prefix=self.prefix)
		#exp.run()
		#exp.solver.file.close()

	def test_load(self):
		s = load_solver(self.path, self.name)
		#s = Experiment.load(self.path, self.name)
		self.assertTrue(isinstance(s, Solver))
		self.assertEqual(s.scheme.__class__.__name__, 'ExplicitEuler')
		self.assertEqual(len(s), 11)
		self.assertEqual(s.name, self.name)
		self.assertEqual(s.store.path, self.path)
		self.assertEqual(s.scheme.h, self.params['scheme_params']['h'])
		with s.open_store() as events:
			self.assertEqual(len(events), 11)
			npt.assert_array_almost_equal(events[-1], np.linspace(0,1,11))

	def test_load_run(self):
		s = load_solver(self.path, self.name)
		s.run(1.)
	
	def test_not_loadable(self):
		base = os.path.dirname(__file__)
		exp_path = os.path.join(base, 'fixtures', 'not_loadable.h5')
		with self.assertWarns(UserWarning):
			s = load_solver(exp_path, 'main')
		len(s) # ensure we can load the events
		# check that solver_info is set:
		info = s.store['solver_info']
		self.assertRegex(info['system_class'], 'NoSystem')
		

	def test_moved(self):
		base = os.path.dirname(__file__)
		exp_path = os.path.join(base, 'fixtures', 'moved.h5')
		s = load_solver(exp_path, 'main')
		len(s)
	
	def test_loadv2(self):
		base = os.path.dirname(__file__)
		exp_path = os.path.join(base,'fixtures', 'format_v2.h5')
		s = load_solver(exp_path, 'tmpexp')
		len(s)


