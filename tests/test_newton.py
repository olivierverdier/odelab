# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from nose.plugins.skip import Skip, SkipTest

from odelab.newton import *

import nose.tools as nt

def no_root(x):
	return 2+np.sin(x)

def test_jacobian():
	def f(x):
		return np.sin(x)
	c = jacobian(f,.5,1e-10)
	nt.assert_almost_equal(c,np.cos(.5),places=6)

class Harness(object):

	dim = 10


	def test_scalar(self):
		def f(x):
			return (x-1)**2
		n = self.solver_class(f)
		n.tol = 1e-9
		z = n.run(10.)
		npt.assert_almost_equal(z,1.)

	def test_copy(self):
		"""Newton doesn't destroy the initial value"""
		def f(x):
			return x
		N = self.solver_class(f)
		x0 = array([1.])
		y0 = x0.copy()
		N.run(x0)
		nt.assert_true(x0 is not y0)
		npt.assert_almost_equal(x0,y0)

	def test_array(self):
		def f(a):
			return a*a
		expected = np.zeros([2,2])
		N = self.solver_class(f)
		x0 = np.ones([2,2])
		res = N.run(x0)
		npt.assert_almost_equal(res, expected)

	@nt.raises(RootSolverDidNotConverge)
	def test_N_no_convergence(self):
		N = self.solver_class(no_root)
		res = N.run(0.)

	def test_complex(self):
		def f(a):
			return a - 1.j
		expected = array([1.j])
		N = self.solver_class(f)
		x0 = 1.+1j
		res = N.run(x0)
		npt.assert_almost_equal(res,expected)


class Test_Newton(Harness):
	solver_class = Newton
	def test_run(self):
		zr = np.random.rand(self.dim)
		def f(x):
			return np.arctan(x - zr)
		xr = np.zeros(self.dim)
		n = self.solver_class(f)
		x = n.run(xr)
		print n.required_iter
		npt.assert_array_almost_equal(x,zr)

class Test_FSolve(Harness):
	solver_class = FSolve
	def test_complex(self):
		raise SkipTest("FSolve does not work with complex")
