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

import pytest

Solver.catch_runtime = False

import scipy.linalg as slin

# test the exponential integrators with a linear problem

class CompareLinearExponential(object):
	def __init__(self, scheme):
		self.description = 'Compare exponential for %s' % type(scheme).__name__
	def __call__(self, computed, expected, phi):
		npt.assert_array_almost_equal(computed, expected)
		npt.assert_array_almost_equal(computed, phi)

@pytest.mark.parametrize("L", [np.array([[1.,2.],[3.,1.]]), -np.identity(2), ])
@pytest.mark.parametrize("scheme",[
			LawsonEuler(),
			RKMK4T(),
			HochOst4(),
			ABLawson2(),
			ABLawson3(),
			ABLawson4(),
			Lawson4(),
			ABNorset4(),
			GenLawson45(),
		]
)
def test_linear_exponential(L, scheme):
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
	CompareLinearExponential(scheme), computed, expected, phi_0

# Complex convection


def check_convection(scheme, time, h, B, u0, sol, do_plot):
	scheme.h = h
	s = SingleStepSolver(scheme, system=B, init_scheme=HochOst4(h=h))
	s.initialize(u0=u0, time=time)
	s.run()
	e1 = s.final()
	u1 = e1[:-1]
	if do_plot:
		pl.plot(B.points, u0)
		pl.plot(B.points, u1)
	npt.assert_array_almost_equal(u1, sol, decimal=2)

schemes = [ExplicitEuler(), 150], [Kutta4(), 10], [Kutta38(), 10], [Heun(), 10], [ABNorset4(), 50], [ABLawson2(), 50], [ABLawson3(), 50], [ABLawson4(), 50], [Lawson4(), 10], [GenLawson45(), 10], [LawsonEuler(), 150], [RKMK4T(), 10], [ode15s(), 2], [AdamsBashforth1(), 150], [AdamsBashforth2(), 50], [AdamsBashforth2e(), 50], [Butcher1(), 50], [Butcher3(), 120],

@pytest.mark.parametrize(["scheme", "N"], schemes, ids=repr)
def test_run(scheme, N, do_plot=False):
	B = BurgersComplex(viscosity=0., size=16)
	umax=.5
	u0 = 2*umax*(.5 - np.abs(B.points))
	time = .5
	mid = time*umax # the peak at final time
	sol = (B.points+.5)*umax/(mid+.5)*(B.points < mid) + (B.points-.5)*umax/(mid-.5)*(B.points > mid)
	if do_plot:
		pl.clf()
		pl.plot(B.points, sol, lw=2)
	check_convection(scheme, time, time/N, B, u0, sol, do_plot)

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


# Auxiliary test to check that the tableaux are square

def check_square(a,b,nb_stages, tail_length):
	npt.assert_equal(len(b), tail_length, 'bv not right # of rows')
	for i,row in enumerate(a):
		npt.assert_equal(len(row), nb_stages+tail_length+1, 'au[%d]'%i)
	for i,row in enumerate(b):
		npt.assert_equal(len(row), nb_stages+tail_length, 'bv[%d]'%i)

import odelab.scheme.exponential as E
exp_objs = [cls() for cls in [getattr(E, name) for name in dir(E)] if hasattr(cls, 'general_linear_z')]

@pytest.mark.parametrize("obj", exp_objs, ids=repr)
def test_exp_square(obj):
	"""
	Check that the matrices produced by the exponential schemes are square.
	"""
	a,b = obj.general_linear_z(np.eye(2))
	nb_stages = len(a)
	tail_length = obj.tail_length
	check_square(a,b, nb_stages, tail_length)
