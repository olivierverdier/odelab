# -*- coding: UTF-8 -*-
from __future__ import division

import numpy.testing as npt
import nose.tools as nt

from odelab.newton import *

class Test_Newton(object):

	dim = 10

	def test_run(self):
		zr = np.random.rand(self.dim)
		def f(x):
			return np.arctan(x - zr)
		xr = np.zeros(self.dim)
		n = Newton(f)
		x = n.run(xr)
		print n.required_iter
		npt.assert_array_almost_equal(x,zr)

	def test_scalar(self):
		def f(x):
			return (x-1)**2
		n = Newton(f)
		n.tol = 1e-9
		z = n.run(10.)
		npt.assert_almost_equal(z,1.)

	def test_copy(self):
		"""Newton doesn't destroy the initial value"""
		def f(x):
			return x
		N = Newton(f)
		x0 = array([1.])
		y0 = x0.copy()
		N.run(x0)
		nt.assert_true(x0 is not y0)
		npt.assert_almost_equal(x0,y0)

	def test_array(self):
		def f(a):
			return a*a
		expected = np.zeros([2,2])
		N = Newton(f)
		x0 = np.ones([2,2])
		res = N.run(x0)
		npt.assert_almost_equal(res, expected)
