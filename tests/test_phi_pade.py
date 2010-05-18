# -*- coding: UTF-8 -*-
from __future__ import division

import scipy.linalg as slin
import numpy.testing as nt

from odelab.phi_pade import *

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
## 		expected = p(Polynomial.exponents(z,1))
		for s in range(1, d+1):
			Z = Polynomial.exponents(z,s)
			computed = p(Z)
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

def compare_phi_pade(computed, expected, phi):
	nt.assert_almost_equal(computed/expected, np.ones_like(expected), decimal=5)
	nt.assert_almost_equal(expected, phi)


def test_phi_pade(k=4,d=10):
	"""
	Test of the Padé approximation of :math:`φ_l` on matrices.
	"""
	phi = Phi(k,d)
	N,D = phi.pade
	for z in  [.1*np.array([[1.,2.],[3.,1.]]),.1j*np.array([[1.j,2.],[3.,1.]]), np.array([[.01]]), np.array([[.1]])]:
		print z
		phis = phi(z)
		for l in range(1,k+1):
			print l
			expected = phi_l(z,l)
			Nz = simple_mul(N[l].coeffs, z)
			Dz = simple_mul(D[l].coeffs, z)
			computed = lin.solve(Dz,Nz)
			yield compare_phi_pade, computed, expected, phis[l]

def compare_to_id(res):
	nt.assert_array_almost_equal(res, np.identity(len(res)))

def test_identity(k=6, d=10):
	"""Test phi_k(0) = Id/k!"""
	z = np.zeros([2,2])
	phi = Phi(k,d)
	phis = phi(z)
	for j,p in enumerate(phis):
		yield compare_to_id, p/phi.C[j]


def test_phi_eval_pade_mat(k=8,d=6):
	z = .1*np.array([[1.,2.],[3.,1.]])
	phi = Phi(k,d)
	phi.eval_pade(z)
	computed = phi.phi[-1]
	expected = phi_l(z,k)

def test_phi_scaled(l=5,d=10):
	z = 100.1
	phi = Phi(l,d)
	expected = phi_l(z,l)
	computed = phi(z)[-1]
	nt.assert_approx_equal(computed, expected)

def test_scaling():
	nt.assert_equal(Phi.scaling(1.), 0)
	nt.assert_equal(Phi.scaling(3.), 2)
	nt.assert_equal(Phi.scaling(.1), 0)

def test_phi_scaled_mat(l=2,d=6):
	A =  np.array([[1.,2.],[3.,1.]])
## 	z = np.random.rand(2,2)
	phi = Phi(l,d)
	for z in [.01*A, .1*A, A, 2*A, 10*A]:
		print z
		expected = phi_l(z,l)
		computed = phi(z)[-1]
		nt.assert_almost_equal(computed/expected, np.ones_like(expected))
		
if __name__ == '__main__':
	test_mat_pol()