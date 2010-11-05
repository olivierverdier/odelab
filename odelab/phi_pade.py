# -*- coding: UTF-8 -*-
"""
:mod:`phi_pade` -- Phi Padé
============================

Computation of φ functions using Padé approximations and Scaling and Squaring.

.. module :: phi_pade
	:synopsis: Computation of φ functions.
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

Some formulae are taken from the `Expint documentation`_, of Håvard Berland, Bård Skaflestad and Will Wright.

.. _Expint project: http://www.math.ntnu.no/num/expint/
.. _Expint documentation: http://www.math.ntnu.no/preprint/numerics/2005/N4-2005.pdf


"""
from __future__ import division

import numpy as np
import numpy.linalg as lin
import math

import sys
sys.py3kwarning = True

def solve(A,b):
	"""
	Silly method to take care of the scalar case.
	"""
	if np.isscalar(A):
		return b/A
	return lin.solve(A,b)

class Polynomial(object):
	"""
Polynomial class used in the Padé approximation.
Usage is as follows::

	p = Polynomial([4.,2.,1.,...]) # polynomial 4 + 2x + x**2 + ...
	Z = Polynomial.exponents(z, 3) # compute z**2 and z**3
	p(Z) # value of the polynomial p(z)

The evaluation at a matrix value is computed using the Paterson and Stockmeyer method (see [Golub]_ § 11.2.4).
The polynomial is split into chunks of size :data:`s`.

.. [Golub] Golub, G.H.  and van Loan, C.F., *Matrix Computations*, 3rd ed. :gbook:`mlOa7wPX6OYC`
	"""
	def __init__(self, coeffs):
		self.coeffs = coeffs

	@classmethod
	def exponents(self, z, s):
		"""
Compute the first s+1 exponents of z.

:Returns:
	[I,z,z**2,...,z**s]
		"""
		if np.isscalar(z):
			ident = 1
		else:
			ident = np.identity(len(z), dtype=z.dtype)
		Z = [ident]
		for i in range(s):
			Z.append(np.dot(Z[-1],z))
		return Z

	def __call__(self, Z):
		"""
Evaluate the polynomial on a matrix, using matrix multiplications (:func:`dot`).

:param list[s+1] Z:
		list of exponents of z up to s, so ``len(Z) == s+1``, where :data:`s` is the size of the chunks
		the various cases are the following:

		:s=1: Horner method
		:s ≥ d: naive polynomial evaluation.
		:s ≈ sqrt(d): optimal choice

		"""
		p = self.coeffs
		P = 0
		s = len(Z) - 1
		if s == 0: # ok only if the polynomial is constant
			if len(p) > 1:
				raise Exception("s may only be zero for constant polynomials")
			return p[0]*Z[0]
		r = int(math.ceil(len(p)/s))
		# assert len(p) <= r*s # this should pass
		for k in reversed(range(r)):
			B = sum(b*Z[j] for j,b in enumerate(p[s*k:s*(k+1)]))
			P = np.dot(Z[s],P) + B
		return P

def ninf(M):
	if np.isscalar(M):
		return abs(M)
	return lin.norm(M,np.inf)

class Phi(object):
	r"""
Main class to compute the :math:`φ_l` functions. 
The simplest way to define those functions is by the formula:

.. math::
	φ_{\ell}(x) = ∑_{k=0}^{∞} \frac{x^k}{(\ell+k)!}

Usage is as follows::

	phi = Phi(k,d)
	result = phi(M)

where :data:`M` is a square array.
The variable ``result`` is a list of all the values of :math:`φ_{k}(M)` for :math:`0≤k≤l`.
	"""

	def __init__(self, k, d=6):
		self.k = k
		self.d = d
		self.pade = self.compute_Pade()

	def compute_Pade(self):
		r"""
Compute the Padé approximations of order :math:`d` of :math:`φ_l`, for :math:`0 ≤ l ≤ k`.

The goal is to produce an approximation of the form:

.. math::
	φ_{\ell} = \frac{N}{D}

where :math:`N` and :math:`D` are polynomials.
The formula for :math:`D` is first computed recursively using the following recursion relations:

.. math::
	D_0^0 = 1\\
	D_{j+1}^0 = \frac{-(d-j)}{(2d -j)(j+1)} D_{j}^0\\
	D_{j}^{\ell+1} = (2d-\ell)(2d+\ell+1-j) D_{j}^{\ell}

Then, considering the inverse factorial series:

.. math::
	C_j := \frac{1}{j!}

The numerator :math:`N` is now computed by:

.. math::
	N = D*C

		"""
		d = self.d
		k = self.k
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
		self.C = C # save for future use; C[j] == 1/j!
		N = [Polynomial(np.convolve(Dr, C[m:m+d+1])[:d+1]) for m,Dr in enumerate(D)]
		return N, [Polynomial(Dl) for Dl in D]

	@classmethod
	def scaling(self, z, threshold=0):
		norm = ninf(z)
		e = max(threshold, np.log2(norm))
		return int(math.ceil(e))

	def eval_pade(self, z, s=None):
		"""
		Evaluate :math:`φ_l(z)` using the Padé approximation.
		"""
		if s is None:
			s = int(math.floor(math.sqrt(self.d)))
		N,D = self.pade
		Z = Polynomial.exponents(z,s)
		self.phi = [solve(PD(Z), PN(Z)) for PN,PD in zip(N,D)]


	def __call__(self, z):
		"""
The final call to compute the values of :math:`φ_k(z)`.
It proceeds in three steps:

1. figure out a scaling exponent :math:`s` such that :math:`z/2^s` is reasonably little
2. compute :math:`φ_k(z/2^s)` using the Padé approximation
3. scale back by repeatedly squaring
		"""
		scaling = self.scaling(z)
		self.eval_pade(z/2**scaling)
		for s in range(scaling):
			self.square()
		return self.phi

	def phi_square(self, l):
		"""
Formula for squaring phi_l from existing phi_k for k≤l, taken from the `Expint documentation`_.

.. _Expint documentation: http://www.math.ntnu.no/preprint/numerics/2005/N4-2005.pdf
		"""
		ifac = self.C
		phi = self.phi
		odd = l % 2
		half = l//2
		next = half
		if odd:
			next += 1
		res = np.dot(phi[half], phi[next])
		res += sum(2*ifac[j]*phi[l-j] for j in xrange(half))
		if odd:
			res += ifac[half]*phi[half+1]
		res /= 2**l
		return res

	def square(self):
		self.phi = [self.phi_square(l) for l in range(self.k+1)]


