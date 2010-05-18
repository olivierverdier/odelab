# -*- coding: UTF-8 -*-
from __future__ import division

from odelab.system import *

import numpy.testing as npt


class Test_ContactOscillator(object):
	def __init__(self, s=3):
		self.s = 3
	def setUp(self):
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
