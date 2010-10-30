# -*- coding: UTF-8 -*-
from __future__ import division

from odelab.system import *

import numpy.testing as npt
import nose.tools as nt

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

def test_Jay_exact(t=1.):
	sys = JayExample()
	dyn = sys.multi_dynamics(np.hstack([JayExample.exact(t,array([1.,1.])),t]))
	res = sum(v for v in dyn.values()) - array([np.exp(t), -2*np.exp(-2*t)])
	npt.assert_array_almost_equal(res,0)

@nt.raises(ValueError)
def test_Jay_exact_wrong_initial_conditions():
	JayExample.exact(1., array([0.,0,0]))

