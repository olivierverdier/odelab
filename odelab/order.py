# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import numpy.linalg
import pylab as pl

import odelab.solver as slv
from odelab.scheme import ode15s

def linear_regression(x, y, do_plot=False):
	from scipy.stats import linregress
	reg = linregress(x, y)
	slope = reg[0]
	if do_plot:
		pl.plot(x,reg[1] + reg[0] * x, linewidth=2., label='slope: %.1f' % slope)
	return reg

class OrderComputer(object):
	"""
	Compute the order of a given system.
	"""

	def __init__(self, solver, system, info, solution):
		self.solver = solver
		self.system = system
		self.info = info
		self.solution = solution

	extra_div = 10 # quotient between minimal step size and reference step size (not used with ode15s)



	def initialize_solver(self, h):
		self.solver.system = self.system
		self.solver.initialize(**self.info)
		self.solver.initialize(h=h)

	def error(self, N):
		h = self.info['time']/N
		self.initialize_solver(h)
		try:
			self.solver.run()
		except (self.solver.Unstable, self.solver.FinalTimeNotReached):
			return None
		return np.log10(np.max(np.abs(self.solver.final()-self.solution)))

	def compute_errors(self, ks):
		Ns = np.round(10**(ks))
		errors = [(N, self.error(N)) for N in Ns]
		self.kerrors = np.array([(np.log10(N),e) for (N,e) in errors if e is not None]).T

	def plot_error(self):
		linear_regression(self.kerrors[0], self.kerrors[1], do_plot=True)
		pl.plot(self.kerrors[0], self.kerrors[1], label=str(self.solver), marker='o')

	def order(self):
		self.reg = linear_regression(self.kerrors[0], self.kerrors[1])
		slope = self.reg[0]
		return slope


class OrderFarm(object):
	def __init__(self, solvers, system, info):
		self.solvers = solvers
		self.system = system
		self.info = info

	def compute_exact(self, ref_solver=None):
		"""
		Compute the exact solution with ode15s.
		"""
		self.ref_solver = slv.SingleStepSolver(ode15s(atol=1.e-15, rtol=5.e-14), self.system)
		self.ref_solver.initialize(**self.info)
		self.ref_solver.initialize(h = self.info['time']/10)
## 		self.ref_solver.initialize(h=self.info['h']/(self.div_coeff**self.max_power)/self.extra_div)
		self.ref_solver.run()
		self.solution = self.ref_solver.final()

	def initialize(self):
		self.compute_exact()
		self.order_computers = [OrderComputer(solver, self.system, self.info, self.solution) for solver in self.solvers]

	def run(self, ks):
		for c in self.order_computers:
			c.compute_errors(ks)

	def orders(self):
		return [(c.solver, c.order()) for c in self.order_computers]

	def plot(self):
		for c in self.order_computers:
			c.plot_error()

