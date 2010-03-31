# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np
import math

class Polynomial(object):
	def __init__(self, coeffs):
		self.coeffs = coeffs

		
	check_partition = True
	def eval_matrix(self, Z, s=1, r=None):
		"""
	Evaluate the polynomial on a matrix, using matrix multiplications (`dot`).

	This is done using the Paterson and Stockmeyer method (see [Golub]_ § 11.2.4).
	The polynomial is split into chunks of size `s`.

	:Parameters:
		Z : list[s+1]
			list of exponents of z up to s, so `len(Z) == s+1`.
		s : int
			size of the chunks; the only limitation is that s ≥ 1; s=1 is the Horner method
			while s ≥ d is the naive polynomial evaluation.
		r : int
			Automatically computed. If given, it should satisfy: :math:`s*r ≥ d+1`

	The number of multiplications is minimised if :math:`s ≈ sqrt(d)`, but
	the choice of s is up to the caller. We explicitly assume :math:`d ≥ s*r`,

	Reference:
.. [Golub] Golub, G.H.  and van Loan, C.F., *Matrix Computations*, 3rd ed..
		"""
		p = self.coeffs
		P = 0
		if r is None:
			r = int(math.ceil(len(p)/s))
		elif self.check_partition: # r is given, we check that it has an acceptable value
			assert len(p) <= r*s
		for k in reversed(range(r)):
			B = sum(b*Z[j] for j,b in enumerate(p[s*k:s*(k+1)]))
			P = np.dot(Z[s],P) + B
		return P

class Pade(object):
	
	def __init__(self, d=6):
		self.d = d
	
	def coefficients(self, k):
		"""
		Compute the Padé approximations of order :math:`d` of :math:`φ_l`, for 0 ≤ l ≤ k.
		"""
		d = self.d
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
		N = [Polynomial(np.convolve(Dr, C[m:m+d+1])[:d+1]) for m,Dr in enumerate(D)]
		return N, [Polynomial(Dl) for Dl in D]


# ==============================================
# Tests
# ==============================================

def Horner(p, x):
	"""
	Numerical value of the polynomial at x
		x may be a scalar or an array
	"""
	def simpleMult(a, b):
		return a*x + b
	return reduce(simpleMult, reversed(p), 0)

def test_mat_pol():
	d = np.random.randint(0,20)
	p = Polynomial(np.random.rand(d+1))
	z = np.random.rand()
	expected = Horner(p.coeffs, z)
	for s in range(1,d):
		Z = [z**j for j in range(s+1)]
		computed = p.eval_matrix(Z, s)
		nt.assert_almost_equal(computed, expected)

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
	pade = Pade(d)
	N,D = pade.coefficients(k)
	for l in range(k):
		expected = phi_l(z,l)
		Nz = Horner(N[l].coeffs, z)
		Dz = Horner(D[l].coeffs, z)
		computed = Nz/Dz
		nt.assert_almost_equal(computed, expected)

		
if __name__ == '__main__':
	test_mat_pol()