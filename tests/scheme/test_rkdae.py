#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import unittest

from odelab.system.graph import QuasiGraphSystem, GraphSystem
from odelab.solver import SingleStepSolver
from odelab.scheme.rungekutta import RKDAE
import odelab.scheme.rungekutta as RK

import numpy as np
import matplotlib.pyplot as plt

import odelab.order as order


def fsin(x):
	return np.sin(x)
fsin.der = np.cos

class Harness_RKDAE(object):
	def setUp(self):
		self.system = QuasiGraphSystem(fsin)
		#self.system = GraphSystem(fsin)
		self.u0 = np.array([0.,0.,1])
		self.set_scheme()
		self.solver = SingleStepSolver(self.scheme, self.system)
		#compare_exact(sol, u0, 2)

	tol = .2 # tolerance when measuring the orders


	def test_quasigraph(self, plot=False):
		sol = self.solver
		errz = []
		errl = []
		ks = np.arange(1,5)
		for k in ks:
			self.scheme.h = pow(2,-k)
			sol.initialize(u0=self.u0,time=1, name='{0}_{1}'.format(type(self).__name__, k))
			sol.run()
			zexact = sol.system.exact(sol.final_time(),self.u0)[0]
			lexact = sol.system.exact(sol.final_time(),self.u0)[2]
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
		self.assertGreater(ol, self.expected_orders[0] - self.tol)
		self.assertGreater(oz, self.expected_orders[1] - self.tol)
		return sol

class Test_LDIRK243(Harness_RKDAE, unittest.TestCase):
	expected_orders = 1,2
	def set_scheme(self):
		self.scheme = RKDAE(tableau=RK.LDIRK343.tableaux[3])
class Test_RadauIIA2(Harness_RKDAE, unittest.TestCase):
	expected_orders = 2,3
	def set_scheme(self):
		self.scheme = RKDAE(tableau=RK.RadauIIA.tableaux[2])
class Test_RadauIIA3(Harness_RKDAE, unittest.TestCase):
	expected_orders = 3,5
	def set_scheme(self):
		self.scheme = RKDAE(tableau=RK.RadauIIA.tableaux[3])
class Test_ImplicitEuler(Harness_RKDAE, unittest.TestCase):
	expected_orders = 1,1
	def set_scheme(self):
		self.scheme = RKDAE(tableau=RK.ImplicitEuler.tableaux[1])
class Test_Gauss(Harness_RKDAE, unittest.TestCase):
	expected_orders = 1,1
	def set_scheme(self):
		self.scheme = RKDAE(tableau=RK.ImplicitEuler.tableaux[1])

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

def test_rkdae():
	sys = GraphSystem(sq)
	u0 = np.array([0.,0.,1.])
	for s in range(2,4):
		scheme = RKDAE(.1, tableau=RK.RadauIIA.tableaux[s])
		sol = SingleStepSolver(scheme, sys)
		sol.initialize(u0=u0, time=1)
		yield CompareExact('RadauIIA-{0}'.format(s)), sol, u0, 2

