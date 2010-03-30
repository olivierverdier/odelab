# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np


def poly_mult(p,x):
	"""
	Numerical value of the polynomial at x
		x may be a scalar or an array
	"""
	def simpleMult(a, b):
		return a*x + b
	return reduce(simpleMult, reversed(p), 0)

class Pade(object):
			
	@classmethod
	def coefficients(self, k, d=6):
		"""
		Compute the Padé approximations of order :math:`d` of :math:`φ_l`, for 0 ≤ l ≤ k.
		"""
		J = np.arange(d+1)
		j = J[:-1]
		a = -(d-j)/(2*d-j)/(j+1)
		D = np.ones([k+1,d+1])
		D[0,1:] = np.cumprod(a)
		l = np.arange(k).reshape(-1,1)
		al = (2*d-l)*(2*d+l+1-J)
		D[1:,:] = np.cumprod(al,0) * D[0,:]
		C = np.empty(d+k+2)
		C[0] = 1.
		C[1:] = 1./np.cumprod(np.arange(d+k+1)+1)
		N = np.array([np.convolve(Dr, C[m:m+d+1])[:d+1] for m,Dr in enumerate(D)])
		return N,D


import numpy.testing as nt

def rec_phi_l(z,l=0):
	"""
	Returns phi_l, 1/l!
	Computes recursively using the recursion formula:
		φ_0 = exp
		φ_{l+1}(z) = \frac{φ_l(z) - \frac{1}{l!}}{z}
	"""
	if l==0:
		return np.exp(z), 1.
	else:
		phi, fac = rec_phi_l(z,l-1)
		newfac = fac/l
		return  (phi - fac)/z, newfac

def phi_l(z, l=0):
	"""
	Compute phi_l using the recursion function `rec_phi_l`
	"""
	return rec_phi_l(z,l)[0]

phi_formulae = { 
	0: lambda z: np.exp(z),
	1: lambda z: np.expm1(z)/z,
	2: lambda z: (np.expm1(z) - z)/z**2,
	3: lambda z: (np.expm1(z) - z - z**2/2)/z**3
}


def test_phi_l():
	"""
	Check that `phi_l` computes :math:`φ_l` correctly (compare to phi_formulae).
	"""
	z = .1
	for l in range(4):
		expected = phi_formulae[l](z)
		computed = phi_l(z, l)
		nt.assert_almost_equal(computed, expected)


def test_phi_pade(k=9,d=10):
	"""
	Test of the Padé approximation of :math:`φ_l`.
	"""
	z = .1
	N,D = Pade.coefficients(k,d)
	for l in range(k):
		expected = phi_l(z,l)
		Nz = poly_mult(N[l],z)
		Dz = poly_mult(D[l],z)
		computed = Nz/Dz
		nt.assert_almost_equal(computed, expected)

		
