# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import pytest

from odelab.system import *
from odelab.system.nonholonomic.contactoscillator import ContactOscillator
from odelab.system.nonholonomic.rolling import VerticalRollingDisk

import numpy.testing as npt

class Test_ContactOscillator(unittest.TestCase):
	def setUp(self):
		self.s = 3
		self.co = ContactOscillator()

	def prepare(self):
		u = np.random.rand(7)
		us = np.column_stack([u]*self.s)
		return u, us

	def test_constraint(self):
		u, us = self.prepare()
		c = self.co.constraint(u)
		cs = self.co.constraint(us)
		npt.assert_array_almost_equal(cs - c.reshape(-1,1), 0)

	def test_reaction_force(self):
		u, us = self.prepare()
		r = self.co.reaction_force(u)
		rs = self.co.reaction_force(us)
		npt.assert_array_almost_equal(rs - r.reshape(-1,1), 0)

	def test_prop_lag(self):
		"""
		reaction force is proportional to lambda
		"""
		n = 5
		u = np.random.rand(7)
		us = np.column_stack([u]*n)
		us[-1] = np.linspace(0,1,5)
		f = self.co.reaction_force(us)
		dfs = np.diff(f)
		npt.assert_array_almost_equal(dfs - dfs[:,0:1], 0)

	def test_dynamics(self):
		u, us = self.prepare()
		d = self.co.multi_dynamics(u)
		ds = self.co.multi_dynamics(us)
		for k in d:
			npt.assert_array_almost_equal(ds[k] - d[k].reshape(-1,1),0)


@pytest.fixture(params=[ContactOscillator(), VerticalRollingDisk()])
def sys(request):
	return request.param

def test_constraint(sys, nb_stages=4):
	u = np.random.random_sample([sys.size,nb_stages])
	constraints = sys.constraint(u)
	for U,C in zip(u.T,constraints.T):
		npt.assert_array_almost_equal(np.dot(sys.codistribution(U),sys.velocity(U)), C)

def test_reaction_force(sys, nb_stages=4):
	u = np.random.random_sample([sys.size,nb_stages])
	force = sys.reaction_force(u)
	for U,F in zip(u.T,force.T):
		npt.assert_array_almost_equal(np.dot(sys.lag(U), sys.codistribution(U),), F)


class TestJay(unittest.TestCase):
	def test_Jay_exact(self, t=1.):
		sys = JayExample()
		dyn = sys.multi_dynamics(np.hstack([JayExample.exact(t,array([1.,1.])),t]))
		res = sum(v for v in dyn.values()) - array([np.exp(t), -2*np.exp(-2*t)])
		npt.assert_array_almost_equal(res,0)

# @nt.raises(ValueError)
	def test_Jay_exact_wrong_initial_conditions(self):
		with self.assertRaises(ValueError):
			JayExample.exact(1., array([0.,0,0]))

