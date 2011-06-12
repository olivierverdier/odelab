#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import odelab.system as dsys
from odelab.solver import SingleStepSolver
from odelab.scheme.constrained import RKDAE
import odelab.scheme.rungekutta as RK

import numpy as np
import pylab as pl

import odelab.order as order

import nose.tools as nt

def fsin(x):
	return np.sin(x)
fsin.der = np.cos

class Harness_RKDAE(object):
	def setUp(self):
		self.system = dsys.QuasiGraphSystem(fsin)
		#self.system = GraphSystem(fsin)
		self.u0 = np.array([0.,0.,1])
		self.set_solver()
		self.solver.auto_save = False
		#compare_exact(sol, u0, 2)

	tol = .2 # tolerance when measuring the orders


	def test_quasigraph(self, plot=False):
		sol = self.solver
		errz = []
		errl = []
		ks = np.arange(1,5)
		for k in ks:
			sol.initialize(u0=self.u0,time=1,h=pow(2,-k), name='{}_{}'.format(type(self).__name__, k))
			sol.run()
			zexact = sol.system.exact(sol.final_time(),self.u0)[0]
			lexact = sol.system.exact(sol.final_time(),self.u0)[2]
			df = sol.final()[0] - zexact
			logerrz = np.log2(np.abs(df))
			logerrl = np.log2(np.abs(sol.final()[2] - lexact))
			errz.append(logerrz)
			errl.append(logerrl)
		pl.clf()
		pl.subplot(1,2,1)
		pl.title('z')
		regz = order.linear_regression(ks,errz,do_plot=True)
		pl.plot(ks,errz,'o-')
		pl.legend()
		pl.subplot(1,2,2)
		pl.title(u'λ')
		regl = order.linear_regression(ks,errl,do_plot=True)
		pl.plot(ks,errl,'o-')
		pl.legend()
		oz = -regz[0]
		ol = -regl[0]
		nt.assert_true(ol > self.expected_orders[0] - self.tol)
		nt.assert_true(oz > self.expected_orders[1] - self.tol)
		return sol

class Test_LDIRK243(Harness_RKDAE):
	expected_orders = 1,2
	def set_solver(self):
		self.solver = SingleStepSolver(RKDAE(RK.LDIRK343.tableaux[3]), self.system)
class Test_RadauIIA2(Harness_RKDAE):
	expected_orders = 2,3
	def set_solver(self):
		self.solver = SingleStepSolver(RKDAE(RK.RadauIIA.tableaux[2]), self.system)
class Test_RadauIIA3(Harness_RKDAE):
	expected_orders = 3,5
	def set_solver(self):
		self.solver = SingleStepSolver(RKDAE(RK.RadauIIA.tableaux[3]),self.system)
class Test_ImplicitEuler(Harness_RKDAE):
	expected_orders = 1,1
	def set_solver(self):
		self.solver = SingleStepSolver(RKDAE(RK.ImplicitEuler.tableaux[1]), self.system)
class Test_Gauss(Harness_RKDAE):
	expected_orders = 1,1
	def set_solver(self):
		self.solver = SingleStepSolver(RKDAE(RK.ImplicitEuler.tableaux[1]), self.system)

if __name__ == '__main__':
	t = Test_LDIRK243()
	t.setUp()
	t.test_quasigraph()
