# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np
from numpy import array, dot
from numpy.linalg import norm, inv
import pylab as PL
from pylab import plot, legend

def jacobian(F,x,h=1e-6):
	"""
	Numerical Jacobian at x.
	"""
	if np.isscalar(x):
		L = 1
	else:
		L = len(x)
	vhs = h * np.identity(L)
	Fs = array([F(x+vh) for vh in vhs])
	grad = (Fs - F(x))/h
	return array(grad).T

class Newton(object):
	"""
	Simple Newton solver to solve F(x) = level.
	"""
	def __init__(self, F=None, level=0.):
		self.F = F
		self.level = level
	
	h = 1e-6
	def der(self, x):
		return jacobian(self.residual, x, self.h)
	
	def residual(self, x):
		a = x.reshape(self.shape)
		res = self.F(a)
		try:
			res_vec = res.ravel()
		except AttributeError: # a list of arrays
			res_vec = np.hstack([comp.ravel() for comp in res])
		return res_vec
	
	maxiter = 600
	tol = 1e-11
	def run(self, x0):
		"""
		Run the Newton iteration.
		"""
		if np.isscalar(x0):
			x = array([x0])
		else:
			x = array(x0)
		self.shape = x.shape
		x = x.ravel()
		for i in xrange(self.maxiter):
			d = self.der(x)
			y = self.level - self.residual(x)
			if np.isscalar(y):
				incr = y/d.item()
			else:
				try:
					incr = np.linalg.solve(d, y)
				except np.linalg.LinAlgError, ex: # should use the "as" syntax!!
					eigvals, eigs = np.linalg.eig(d)
					zerovecs = eigs[:, np.abs(eigvals) < 1e-10]
					raise np.linalg.LinAlgError("%s: %s" % (ex.message, repr(zerovecs)))
			x += incr
			if norm(incr) < self.tol:
				break
##			if self.is_zero(x):
##				break
		else:
			raise Exception("Newton algorithm did not converge. ∆x=%.2e" % norm(incr))
		self.required_iter = i
		return x.reshape(self.shape)
	
	def is_zero(self, x):
		res = norm(self.F(x) - self.level)
		return res < self.tol
