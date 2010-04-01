# -*- coding: UTF-8 -*-
"""
:mod:`phi_pade` -- Phi Padé
============================

Computation of φ functions using Padé approximations and Scaling and Squaring.

.. module :: phi_pade
	:synopsis: Computation of φ functions.
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

"""
from __future__ import division

import numpy as np
import numpy.linalg as lin
import math

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
	"""
	def __init__(self, coeffs):
		self.coeffs = coeffs
	
	@classmethod
	def exponents(self, z, s):
		"""
		Compute the first s+1 exponents of z		
		"""
		if np.isscalar(z):
			ident = 1
		else:
			ident = np.identity(len(z))
		Z = [ident]
		for i in range(s):
			Z.append(np.dot(Z[-1],z))
		return Z
		
	def eval_matrix(self, Z):
		"""
Evaluate the polynomial on a matrix, using matrix multiplications (:func:`dot`).

This is done using the Paterson and Stockmeyer method (see [Golub]_ § 11.2.4).
The polynomial is split into chunks of size :data:`s`.

:Parameters:
	Z : list[s+1]
		list of exponents of z up to s, so ``len(Z) == s+1``, where :data:`s` is the size of the chunks;
		s=1, it is the Horner method
		s ≥ d is the naive polynomial evaluation.
		s ≈ sqrt(d) is the optimal choice

:Reference:

.. [Golub] Golub, G.H.  and van Loan, C.F., *Matrix Computations*, 3rd ed..
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
	"""
Main class to compute the :math:`φ_l` functions. The simplest way to define those functions is:

.. math::
	φ_l = ∑_{k=0}^{∞} \frac{x^k}{(l+k)!}

Usage is as follows::

	phi = Phi(k,d)
	phi.eval_matrix(M)

where :data:`M` is a square array.
	"""
	
	def __init__(self, k, d=6):
		self.k = k
		self.d = d
		self.pade = self.compute_Pade()
	
	def compute_Pade(self):
		"""
		Compute the Padé approximations of order :math:`d` of :math:`φ_l`, for :math:`0 ≤ l ≤ k`.
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
	
	def scaling(self, z):
		norm = ninf(z)
		return int(math.ceil(math.log(norm,2)))
	
	def eval_pade(self, z, s=None):
		"""
		Evaluate :math:`φ_l(z)` using the Padé approximation.
		"""
		if s is None:
			s = int(math.floor(math.sqrt(self.d)))
		N,D = self.pade
		Z = Polynomial.exponents(z,s)
		self.phi = [solve(PD.eval_matrix(Z), PN.eval_matrix(Z)) for PN,PD in zip(N,D)]
	
	
	def eval(self, z):
		scaling = self.scaling(z)
		print 'scaling', scaling
		scaled_eval = self.eval_pade(z/2**scaling)
		for s in range(scaling):
			self.square()
		return self.phi[-1]
	
	def square(self):
		ifac = self.C
		phi = self.phi
		newphi = []
		for l in range(self.k+1):
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
			newphi.append(res)
		self.phi = newphi
		

# ==============================================
# Tests
# ==============================================

import scipy.linalg as slin
import numpy.testing as nt

def test_poly_exps():
	x = np.array([[1.,2.],[3.,1.]])
	x2 = np.dot(x,x)
	X = Polynomial.exponents(x,2)
	nt.assert_array_almost_equal(X[-1],x2)
	nt.assert_array_almost_equal(X[1],x)
	nt.assert_array_almost_equal(X[0], np.identity(2))


def simple_mul(p, x):
	"""
	Numerical value of the polynomial at x
		x may be a scalar or an array
	"""
	X = Polynomial.exponents(x,len(p)-1)
	return sum(pk*xk for pk,xk in zip(p,X))

def test_simple_mul_mat():
	X = np.array([[1.,2.],[3.,1.]])
	expected = 9.*np.identity(2) + 3.*X + 2.*np.dot(X,X)
	computed = simple_mul([9.,3.,2.], X)
	nt.assert_almost_equal(computed, expected)


def test_mat_pol(n=2):
	for d in range(1,20):
		p = Polynomial(np.random.rand(d+1))
		z = np.random.rand(n,n)
		expected = simple_mul(p.coeffs, z)
## 		expected = p.eval_matrix(Polynomial.exponents(z,1))
		for s in range(1, d+1):
			Z = Polynomial.exponents(z,s)
			computed = p.eval_matrix(Z)
			print p.coeffs, s
			nt.assert_almost_equal(computed, expected)



def expm(M):
	"""
	Matrix exponential from :mod:`scipy`; adapt it to work on scalars.
	"""
	if np.isscalar(M):
		return math.exp(M)
	else:
		return slin.expm(M)


def phi_l(z, l=0):
	"""
	Returns phi_l using the recursion formula:
		φ_0 = exp
		φ_{l+1}(z) = \frac{φ_l(z) - \frac{1}{l!}}{z}
	"""
	phi = expm(z)
	fac = Polynomial.exponents(z,0)[0] # identity
	if np.isscalar(z):
		iz = 1./z
	else:
		iz = lin.inv(z)
	for i in range(l):
		phi =  np.dot(phi - fac, iz)
		fac /= i+1
	return phi

phi_formulae = { 
	0: lambda z: np.exp(z),
	1: lambda z: np.expm1(z)/z,
	2: lambda z: (np.expm1(z) - z)/z**2,
	3: lambda z: (np.expm1(z) - z - z**2/2)/z**3
}


def test_phi_l():
	"""
	Check that :func:`phi_l` computes :math:`φ_l` correctly for scalars (by comparing to :data:`phi_formulae`).
	"""
	z = .1
	for l in range(4):
		expected = phi_formulae[l](z)
		computed = phi_l(z, l)
		nt.assert_almost_equal(computed, expected)

def test_phi_0_mat():
	z = np.random.rand(2,2)
	expected = expm(z)
	computed = phi_l(z,0)
	nt.assert_almost_equal(computed, expected)

def test_phi_1_mat():
	z = np.random.rand(2,2)
	expected = expm(z) - np.identity(2)
	expected = lin.solve(z, expected)
	computed = phi_l(z,1)
	nt.assert_almost_equal(computed, expected)

def test_phi_pade(k=8,d=10):
	"""
	Test of the Padé approximation of :math:`φ_l` on matrices.
	"""
	z = .1*np.array([[1.,2.],[3.,1.]])
	phi = Phi(k,d)
	N,D = phi.pade
	for l in range(k):
		expected = phi_l(z,l)
		Nz = simple_mul(N[l].coeffs, z)
		Dz = simple_mul(D[l].coeffs, z)
		computed = lin.solve(Dz,Nz)
		nt.assert_almost_equal(computed, expected)

def test_phi_eval_pade(k=8,d=6):
	z = .1*np.array([[1.,2.],[3.,1.]])
	phi = Phi(k,d)
	phi.eval_pade(z)
	computed = phi.phi[-1]
	expected = phi_l(z,k)

def test_phi_scaled(l=5,d=10):
	z = 100.1
	phi = Phi(l,d)
	expected = phi_l(z,l)
	computed = phi.eval(z)
	nt.assert_approx_equal(computed, expected)


def test_phi_scaled_mat(l=2,d=6):
	z = np.array([[1.,2.],[3.,1.]])
## 	z = np.random.rand(2,2)
	phi = Phi(l,d)
	expected = phi_l(z,l)
	computed = phi.eval(z)
	nt.assert_almost_equal(computed, expected)
		
if __name__ == '__main__':
	test_mat_pol()