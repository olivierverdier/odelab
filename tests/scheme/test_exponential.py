#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.scheme import *
from odelab.scheme.exponential import *
from odelab.scheme.rungekutta import *
from odelab.scheme.generallinear import *
from odelab.scheme.classic import *

from odelab.system import *
from odelab.solver import *

import numpy.testing as npt

from nose.plugins.skip import SkipTest


Solver.catch_runtime = False

import scipy.linalg as slin

# test the exponential integrators with a linear problem

class CompareLinearExponential(object):
	def __init__(self, scheme):
		self.description = 'Compare exponential for %s' % type(scheme).__name__
	def __call__(self, computed, expected, phi):
		npt.assert_array_almost_equal(computed, expected)
		npt.assert_array_almost_equal(computed, phi)

def test_linear_exponential():
	for L in [np.array([[1.,2.],[3.,1.]]), -np.identity(2), ]: # np.zeros([2,2])
		for scheme in [
			LawsonEuler(),
			RKMK4T(),
			HochOst4(),
			ABLawson2(),
			ABLawson3(),
			ABLawson4(),
			Lawson4(),
			ABNorset4(),
			GenLawson45(),
		]:
			h = .1
			scheme.h = h
			sys = Linear(L)
			s = SingleStepSolver(scheme, system=sys, init_scheme=HochOst4(h=h))
			u0 = np.array([1.,0.])
			s.initialize(u0 = u0)

			s.run(time=1.)
			computed = s.final()[:-1]
			phi = Phi(0)
			tf = s.final_time()
			phi_0 = np.dot(phi(tf*L)[0], u0)
			expected = np.dot(slin.expm(tf*L), u0)
			yield CompareLinearExponential(scheme), computed, expected, phi_0

# Complex convection

class Harness_ComplexConvection(object):

	def check_convection(self, do_plot):
		scheme = self.scheme
		h = self.time/self.N
		scheme.h = h
		self.s = SingleStepSolver(scheme, system=self.B, init_scheme=HochOst4(h=h))
		self.s.initialize(u0=self.u0, time=self.time)
		self.s.run()
		e1 = self.s.final()
		u1 = e1[:-1]
		if do_plot:
			pl.plot(self.B.points, self.u0)
			pl.plot(self.B.points, u1)
		npt.assert_array_almost_equal(u1, self.sol, decimal=2)

	def test_run(self, do_plot=False):
		self.B = BurgersComplex(viscosity=0., size=32)
		umax=.5
		self.u0 = 2*umax*(.5 - np.abs(self.B.points))
		self.time = .5
		mid = self.time*umax # the peak at final time
		self.sol = (self.B.points+.5)*umax/(mid+.5)*(self.B.points < mid) + (self.B.points-.5)*umax/(mid-.5)*(self.B.points > mid)
		if do_plot:
			pl.clf()
			pl.plot(self.B.points, self.sol, lw=2)
		self.check_convection(do_plot)

	def find_N(self):
		for N in [10,20,50,75,100,120,150]:
			self.N = N
			try:
				self.notest_run()
			except AssertionError:
				continue
			else:
				break
		else:
			raise Exception('No N!')
		print(type(self.scheme).__name__, N)

class Test_CC_EE(Harness_ComplexConvection):
	scheme = ExplicitEuler()
	N=150

class Test_CC_RK4(Harness_ComplexConvection):
	scheme = Kutta4()
	N = 10

class Test_CC_RK38(Harness_ComplexConvection):
	scheme = Kutta38()
	N = 10

class Test_CC_Heun(Harness_ComplexConvection):
	scheme = Heun()
	N = 10

class Test_CC_ABN4(Harness_ComplexConvection):
	scheme = ABNorset4()
	N = 50

class Test_CC_ABL2(Harness_ComplexConvection):
	scheme = ABLawson2()
	N = 50

class Test_CC_ABL3(Harness_ComplexConvection):
	scheme = ABLawson3()
	N=50

class Test_CC_ABL4(Harness_ComplexConvection):
	scheme = ABLawson4()
	N=50

class Test_CC_L4(Harness_ComplexConvection):
	scheme = Lawson4()
	N=10

class Test_CC_GL45(Harness_ComplexConvection):
	scheme = GenLawson45()
	N=10

class Test_CC_LE(Harness_ComplexConvection):
	scheme = LawsonEuler()
	N=150

class Test_CC_RKMK4T(Harness_ComplexConvection):
	scheme = RKMK4T()
	N=10

class Test_CC_ode15s(Harness_ComplexConvection):
	scheme = ode15s()
	N=2

class Test_CC_AB1(Harness_ComplexConvection):
	scheme = AdamsBashforth1()
	N = 150

class Test_CC_AB2(Harness_ComplexConvection):
	scheme = AdamsBashforth2()
	N = 50

class Test_CC_AB2e(Harness_ComplexConvection):
	scheme = AdamsBashforth2e()
	N = 50

class Test_CC_B1(Harness_ComplexConvection):
	scheme = Butcher1()
	N=50

class Test_CC_B3(Harness_ComplexConvection):
	scheme = Butcher3()
	N=120

# Auxiliary test to check that the tableaux are square

class CheckSquare(object):
	def __init__(self, name):
		self.description = name
	def __call__(self,name,a,b,nb_stages, tail_length):
		npt.assert_equal(len(b), tail_length, 'bv not right # of rows')
		for i,row in enumerate(a):
			npt.assert_equal(len(row), nb_stages+tail_length+1, 'au[%d]'%i)
		for i,row in enumerate(b):
			npt.assert_equal(len(row), nb_stages+tail_length, 'bv[%d]'%i)

def test_exp_square():
	"""
	Check that the matrices produced by the exponential schemes are square.
	"""
	import odelab.scheme.exponential as E
	for name in dir(E):
		cls = getattr(E, name)
		if hasattr(cls, 'general_linear_z'):
			obj = cls()
			a,b = obj.general_linear_z(np.eye(2))
			nb_stages = len(a)
			tail_length = obj.tail_length
			yield CheckSquare(name),name, a,b, nb_stages, tail_length


