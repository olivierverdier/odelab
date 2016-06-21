# -*- coding: utf-8 -*-
from __future__ import division

from odelab.order import *
from odelab.solver import *
from odelab.scheme.exponential import *
from odelab.system import *

class Order_Burgers(object):
	def test_run(self):
		B = BurgersComplex(viscosity=.03)
		size = B.size
		u0 = .5 - np.abs(B.points)
## 		u0 = np.cos(B.points*(2*np.pi))*(1+np.sin(B.points*(2*np.pi)))
		info = {'u0':u0, 't0':0., 'time':.5, }

		solvers = [MultiStepSolver(scheme_class(), B) for scheme_class in [LawsonEuler, RKMK4T, HochOst4, GenLawson45,ABNorset4,Lawson4,ABLawson4][3:4]]
		self.o = OrderFarm(solvers, B, info)
		self.o.initialize()
		self.o.run(np.linspace(2,3,5))
		print(self.o.orders())
		self.o.plot()


def make_lin(A):
	if np.isscalar(A):
		def lin(t,u):
			return A*u
	else:
		def lin(t, u):
			return dot(A,u)
	lin.exact = make_exp(A)
	return lin

def make_exp(A):
	if np.isscalar(A):
		def exact(u0,t0,t):
			return u0 * exp((t-t0)*A)
	else:
		def exact(u0, t0, t):
			return dot(expm((t-t0)*A),u0)
	return exact

class Harness_Solver_Order(object):
	no_plot = True
	a = -1.
	u0 = 1.
	time = 1.
	do_plot=False
	def notest_order(self):
		self.solver.initialize(u0=self.u0, time=self.time)
		order = self.solver.plot_error(do_plot=self.do_plot)
		print(order)
		nt.assert_true(order < self.order + .1)

class Test_ExplicitEuler(Harness_Solver_Order):
	def setUp(self):
		self.solver = Solver(ExplicitEuler(h=.1), System(make_lin(self.a)))
		self.order = -1.

class Test_ImplicitEuler(Harness_Solver_Order):
	def setUp(self):
		self.solver = Solver(ImplicitEuler(h=.1), System(make_lin(self.a)))
		self.order = -1.

class Test_RungeKutta4(Harness_Solver_Order):
	def setUp(self):
		self.solver = Solver(RungeKutta4(h=.1), System(make_lin(self.a)))
		self.solver.err_kmin = 1
		self.solver.err_kmax = 2.5
		self.order = -4.
