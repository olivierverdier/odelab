# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np

from odelab.scheme import Scheme

from odelab.phi_pade import Phi, Polynomial

class Exponential(Scheme):
	"""
	Explicit Exponential Integrator Class.
	"""
	def __init__(self, *args, **kwargs):
		super(Exponential, self).__init__(*args, **kwargs)
		self.phi = Phi(self.phi_order, self.phi_degree)
	
	phi_degree = 6
	
	def initialize(self):
		super(Exponential, self).initialize()
		ts = self.solver.ts[-self.tail_length:]
		tail = self.solver.us[-self.tail_length:]
		# this is specific to those Exponential solvers:
		for i in range(len(tail)-1):
			tail[i] = self.h*self.system.nonlin(ts[i], tail[i])
		self.tail = np.array(list(reversed(tail))).T
	
	def step(self, t, u):
		h = self.h
		ua, vb = self.general_linear()
		nb_stages = len(ua)
		nb_steps = len(vb)
		Y = np.zeros([len(u), nb_stages+nb_steps], dtype=u.dtype)
		Y[:,-nb_steps:] = self.tail
		newtail = np.zeros_like(self.tail) # alternative: work directly on self.tail
		for s in range(nb_stages):
			uas = ua[s]
			for j, coeff in enumerate(uas[1:]):
				if coeff is not None:
					Y[:,s] += np.dot(coeff, Y[:,j])
			Y[:,s] = h*self.system.nonlin(t+uas[0]*h, Y[:,s])
		for r in range(nb_steps):
			vbr = vb[r]
			for j, coeff in enumerate(vbr):
				if coeff is not None:
					newtail[:,r] += np.dot(coeff, Y[:,j])
		self.tail = newtail
		return t + h, self.tail[:,0]
		

	def general_linear(self):
		"""
		Currently returns a cached version of the coefficients of the method.
		"""
		if self._h_dirty: # recompute the coefficients if h had changed
			z = self.h * self.system.linear()
			self._general_linear = self.general_linear_z(z)
			self._h_dirty = False
		return self._general_linear

class LawsonEuler(Exponential):
	phi_order = 0
	
	def general_linear_z(self, z):
		ez = self.phi(z)[0]
		one = Polynomial.exponents(z,0)[0]
		return [[0., None, one]], [[ez, ez]]

class RKMK4T(Exponential):
	phi_order = 1
	
	def general_linear_z(self, z):
		one = Polynomial.exponents(z,0)[0]
		ez, phi_1 = self.phi(z)
		ez2, phi_12 = self.phi(z/2)
		return ([	[0, None, None, None, None, one],
					[1/2, 1/2*phi_12, None, None, None, ez2],
					[1/2, 1/8*np.dot(z, phi_12), 1/2*np.dot(phi_12,one-1/4*z), None, None, ez2],
					[1, None, None, phi_1, None, ez]
				],
				[[1/6*np.dot(phi_1,one+1/2*z), 1/3*phi_1, 1/3*phi_1, 1/6*np.dot(phi_1,one-1/2*z), ez]]
				)
		

class HochOst4(Exponential):
	phi_order = 3
	phi_degree = 13
	
	def general_linear_z(self, z):
		one = Polynomial.exponents(z,0)[0]
		ez, phi_1, phi_2, phi_3 = self.phi(z)
		ez2, phi_12, phi_22, phi_32 = self.phi(z/2)
		a_52 = 1/2*phi_22 - phi_3 + 1/4*phi_2 - 1/2*phi_32
		a_54 = 1/4*phi_22 - a_52
		return ([	[0, None, None, None, None, None, one],
					[1/2, 1/2*phi_12, None, None, None, None, ez2],
					[1/2, 1/2*phi_12 - phi_22, phi_22, None, None, None, ez2],
					[1, phi_1-2*phi_2, phi_2, phi_2, None, None, ez],
					[1/2, 1/2*phi_12 - 2*a_52 - a_54, a_52, a_52, a_54, None, ez2]
				],
				[	[phi_1 - 3*phi_2 + 4*phi_3, None, None, -phi_2 + 4*phi_3, 4*phi_2 - 8*phi_3, ez],
				])

class ABLawson2(Exponential):
	phi_order = 2
	tail_length = 2
	
	def general_linear_z(self, z):
		one = Polynomial.exponents(z,0)[0]
		ez, phi_1, phi_2 = self.phi(z)
		ez2 = self.phi(z/2)[0]
		e2z = np.dot(ez,ez)
		return ([	[0, None, None, one, None],
					[1., 3/2*ez, None, ez, -1/2*e2z]
				],
				[	[3/2*ez, None, ez, -1/2*ez2],
					[one, None, None, None]
				])
