# -*- coding: UTF-8 -*-
"""
:mod:`exponential`

The numerical values of this exponential scheme are taken from the code of the `Expint package`_.
The peculiarities of the :meth:`tail` method are explained and argumented in the `Expint documentation`_.

.. _Expint project: http://www.math.ntnu.no/num/expint/
.. _Expint documentation: http://www.math.ntnu.no/preprint/numerics/2005/N4-2005.pdf

.. module:: exponential
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

"""
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
		tail = np.array(self.solver.events[:,-self.tail_length:])
		# this is specific to those Exponential solvers:
		# warning: this creates a tail using the type of u
		for i in range(tail.shape[1]-1):
			# if nonlin returns complex and u is float, type cast is performed:
			tail[:-1,i] = self.h*self.system.nonlin(tail[-1,i], tail[:-1,i])
		self.tail = np.array(tail[:-1,::-1])

	def step(self, t, u):
		h = self.h
		ua, vb = self.general_linear()
		nb_stages = len(ua)
		nb_steps = len(vb)
		Y = np.zeros([len(u), nb_stages+nb_steps], dtype=self.tail.dtype)
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

class ABLawson(Exponential):
	phi_order = 0

class ABLawson2(ABLawson):
	tail_length = 2

	def general_linear_z(self, z):
		ez = self.phi(z)[0]
		one, ez, e2z = Polynomial.exponents(ez,2)
		ez2 = self.phi(z/2)[0]
		return ([	[0, None, None, one, None],
					[1., 3/2*ez, None, ez, -1/2*e2z]
				],
				[	[3/2*ez, None, ez, -1/2*ez2],
					[one, None, None, None]
				])

class ABLawson3(ABLawson):
	phi_order = 0
	tail_length = 3

	def general_linear_z(self, z):
		ez = self.phi(z)[0]
		one, ez, e2z, e3z = Polynomial.exponents(ez,3)
		return ([	[0, None, None, one, None, None],
					[1., 23/12*ez, None, ez,  -4/3*e3z,  5/12*e3z]
				],
				[	[23/12*ez, None, ez, -4/3*e2z, 5/12*e3z],
					[one, None, None, None, None],
					[None, None, None, one, None],
				])

class ABLawson4(ABLawson):
	phi_order = 0
	tail_length = 4

	def general_linear_z(self, z):
		ez = self.phi(z)[0]
		one, ez, e2z, e3z, e4z = Polynomial.exponents(ez, 4)
		return ([	[0, None, None, one, None, None, None],
					[1, 55/24*ez, None, ez, -59/24*e2z, 37/24*e3z, -3/8*e4z]
				],
				[	[55/24*ez, None, ez, -59/24*e2z, 37/24*e3z, -3/8*e4z],
					[one, None, None, None, None, None],
					[None, None, None, one, None, None],
					[None, None, None, None, one, None]
				])

class Lawson4(Exponential):
	phi_order = 0
	tail_length = 1

	def general_linear_z(self, z):
		one = Polynomial.exponents(z,0)[0]
		ez = self.phi(z)[0]
		ez2 = self.phi(z/2)[0]
		return ([	[0, None, None, None, None, one],
					[1/2, 1/2*ez2, None, None, None, ez2],
					[1/2, None, 1/2*one, None, None, ez2],
					[1., None, None, ez2, None, ez]
				],
				[[1/6*ez, 1/3*ez2, 1/3*ez2, 1/6*one, ez]])

class ABNorset4(Exponential):
	phi_order = 4
	tail_length = 4

	def general_linear_z(self, z):
		ez, phi_1, phi_2, phi_3, phi_4 = self.phi(z)
		one = Polynomial.exponents(z,0)[0]
		return ([	[0, None, None, one, None, None, None],
					[1., phi_1 + 11/6*phi_2 + 2*phi_3 + phi_4, None, ez,   -3*phi_2 - 5*phi_3 - 3*phi_4, 3/2*phi_2 + 4*phi_3 + 3*phi_4,   -1/3*phi_2 - phi_3 - phi_4 ]
				],
				[	[phi_1 + 11/6*phi_2 + 2*phi_3 + phi_4, None, ez, -3*phi_2 - 5*phi_3 - 3*phi_4, 3/2*phi_2 + 4*phi_3 + 3*phi_4, -1/3*phi_2 - phi_3 - phi_4],
					[one, None, None, None, None, None],
					[None, None, None, one, None, None],
					[None, None, None, None, one, None],
				])

class GenLawson45(Exponential):
	phi_order = 5
	tail_length = 5

	def general_linear_z(self, z):
		one = Polynomial.exponents(z,0)[0]
		ez, phi_1, phi_2, phi_3, phi_4, phi_5 = self.phi(z)
		ez2, phi_12, phi_22, phi_32, phi_42, phi_52 = self.phi(z/2)
		return ([	[0,None,None,None,None,
						one,None,None,None,None],
					[1/2,1/2*phi_12 + 25/48*phi_22 + 35/96*phi_32 + 5/32*phi_42 + 1/32*phi_52, None,None,None,
						ez2, -phi_22 - 13/12*phi_32 - 9/16*phi_42 - 1/8*phi_52, 3/4*phi_22 + 19/16*phi_32 + 3/4*phi_42 + 3/16*phi_52, -1/3*phi_22 - 7/12*phi_32 - 7/16*phi_42 - 1/8*phi_52, 1/16*phi_22 + 11/96*phi_32 + 3/32*phi_42 + 1/32*phi_52,],
					[1/2,1/2*phi_12 + 25/48*phi_22 + 35/96*phi_32 + 5/32*phi_42 + 1/32*phi_52 - 315/256*one, 1/2*one,None,None,
						ez2, -phi_22 - 13/12*phi_32 - 9/16*phi_42 - 1/8*phi_52 + 105/64*one, 3/4*phi_22 + 19/16*phi_32 + 3/4*phi_42 + 3/16*phi_52 - 189/128*one,   -1/3*phi_22 - 7/12*phi_32 - 7/16*phi_42 - 1/8*phi_52 + 45/64*one,       1/16*phi_22 + 11/96*phi_32 + 3/32*phi_42 + 1/32*phi_52 - 35/256*one],
					[1.,phi_1 + 25/12*phi_2 + 35/12*phi_3 + 5/2*phi_4 + phi_5 - 315/128*ez2, None,
						ez2, None,  ez, -4*phi_2 - 26/3*phi_3 - 9*phi_4 - 4*phi_5 + 105/32*ez2,  3*phi_2 + 19/2*phi_3 + 12*phi_4 + 6*phi_5 - 189/64*ez2,      -4/3*phi_2 - 14/3*phi_3 - 7*phi_4 - 4*phi_5 + 45/32*ez2,       1/4*phi_2 + 11/12*phi_3 + 3/2*phi_4 + phi_5 - 35/128*ez2],],
				[	[phi_1 + 25/12*phi_2 + 35/12*phi_3 + 5/2*phi_4 + phi_5 - 5/6*one - 105/64*ez2, 1/3*ez2, 1/3*ez2, 1/6*one,
							ez,  -4*phi_2 - 26/3*phi_3 - 9*phi_4 - 4*phi_5 + 5/3*one + 35/16*ez2, 3*phi_2 + 19/2*phi_3 + 12*phi_4 + 6*phi_5 - 5/3*one - 63/32*ez2, -4/3*phi_2 - 14/3*phi_3 - 7*phi_4 - 4*phi_5 + 5/6*one + 15/16*ez2, 1/4*phi_2 + 11/12*phi_3 + 3/2*phi_4 + phi_5 - 1/6*one - 35/192*ez2],
					[one, None,None,None,None,
						None,None,None,None,],
					[None,None,None,None,
						None,one,None,None,None],
					[None,None,None,None,
						None,None,one,None,None],
					[None,None,None,None,
						None,None,None,one,None],
				],
				)

