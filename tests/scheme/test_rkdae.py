#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import unittest
import pytest

from odelab.system.graph import QuasiGraphSystem, GraphSystem
from odelab.solver import Solver
from odelab.scheme.rungekutta import RKDAE
import odelab.scheme.rungekutta as RK

import numpy as np
import matplotlib.pyplot as plt

import odelab.order as order


def fsin(x):
	return np.sin(x)
fsin.der = np.cos

@pytest.mark.parametrize(["expected_orders", "scheme"], 
[
		((1,2), RKDAE(tableau=RK.LDIRK343.tableaux[3]),),
		((2,3), RKDAE(tableau=RK.RadauIIA.tableaux[2]),),
		((3,5), RKDAE(tableau=RK.RadauIIA.tableaux[3]),),
		((1,1), RKDAE(tableau=RK.ImplicitEuler.tableaux[1]),),
		((1,1), RKDAE(tableau=RK.ImplicitEuler.tableaux[1]),),
], ids=repr)
def test_orders(expected_orders, scheme, plot=False, tol=.2):
	system = QuasiGraphSystem(fsin)
	#system = GraphSystem(fsin)
	u0 = np.array([0.,0.,1])
	solver = Solver(scheme, system)
	#compare_exact(sol, u0, 2)

	sol = solver
	errz = []
	errl = []
	ks = np.arange(1,5)
	for k in ks:
		scheme.h = pow(2,-k)
		sol.initialize(u0=u0)
		sol.run(1)
		zexact = sol.system.exact(sol.final_time(),u0)[0]
		lexact = sol.system.exact(sol.final_time(),u0)[2]
		df = sol.final()[0] - zexact
		logerrz = np.log2(np.abs(df))
		logerrl = np.log2(np.abs(sol.final()[2] - lexact))
		errz.append(logerrz)
		errl.append(logerrl)
	if plot:
		plt.clf()
		plt.subplot(1,2,1)
		plt.title('z')
		plt.plot(ks,errz,'o-')
		plt.legend()
		plt.subplot(1,2,2)
		plt.title(u'λ')
		plt.plot(ks,errl,'o-')
		plt.legend()
	regz = order.linear_regression(ks,errz,do_plot=False)
	regl = order.linear_regression(ks,errl,do_plot=False)
	oz = -regz[0]
	ol = -regl[0]
	assert ol > expected_orders[0] - tol
	assert oz > expected_orders[1] - tol
	return sol



# RK DAE

class CompareExact(object):
	def __init__(self, name):
		self.description = name
	def __call__(self, solver, u0, components, decimal=2):
		solver.run()
		print(solver.final_time())
		print(solver.final())
		exact = solver.system.exact(solver.final_time(), u0)
		#npt.assert_array_almost_equal(solver.final()[:components], exact[:components], decimal=decimal)

def sq(x):
	return .5*x*x
def lin(x):
	return x
sq.der = lin

@pytest.mark.parametrize('s', range(2,4))
def test_rkdae(s):
	sys = GraphSystem(sq)
	u0 = np.array([0.,0.,1.])
	scheme = RKDAE(.1, tableau=RK.RadauIIA.tableaux[s])
	sol = Solver(scheme, sys)
	sol.initialize(u0=u0, time=1)
	CompareExact('RadauIIA-{0}'.format(s)), sol, u0, 2

