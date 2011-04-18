# -*- coding: UTF-8 -*-
from __future__ import division

from odelab.scheme import *
from odelab.scheme.constrained import *

from odelab.system import *
from odelab.solver import *

import numpy.testing as npt


SingleStepSolver.catch_runtime = False
SingleStepSolver.auto_save = True
SingleStepSolver.shelf_name = 'bank_constrained'

# Contact oscillator

class Harness_Osc(object):
	def setUp(self):
		self.sys = ContactOscillator()
		self.set_solver()
		self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
		self.s.time = 10.

## 	def test_run(self):
## 		self.s.run()


	z0s = np.linspace(-.9,.9,10)
	N = 40

	def test_z0(self, i=5, nb_Poincare_iterations=10):
		z0 = self.z0s[i]
		h = self.sys.time_step(self.N)
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial(z0), h=h, time=time)
		self.s.run()
		npt.assert_almost_equal(self.sys.energy(self.s.final()), self.sys.energy(self.s.initial()), decimal=1)

	def plot_qv(self, i=2, skip=None, *args, **kwargs):
		if skip is None:
			skip = self.N
		qs = self.sys.position(self.s.aus)
		vs = self.sys.velocity(self.s.aus)
		if not kwargs.get('marker') and not kwargs.get('ls'):
			kwargs['ls'] = ''
			kwargs['marker'] = 'o'
		plot(qs[i,::skip], vs[i,::skip], *args, **kwargs)


class Test_McOsc(Harness_Osc):
	def set_solver(self):
		self.s = SingleStepSolver(McLachlan(), self.sys)

class Test_JayOsc(Harness_Osc):
	N=5 # bigger time step to make test faster
	def set_solver(self):
		self.s = SingleStepSolver(Spark(2), self.sys)

class Test_HOsc(Harness_Osc):
	N=5 # bigger time step to make test faster
	def set_solver(self):
		self.s = SingleStepSolver(NonHolonomicEnergy(), self.sys)

# Vertical Rolling Disk

class Harness_VerticalRollingDisk(object):
	def setUp(self):
		self.sys = VerticalRollingDisk()
		self.setup_solver()
		ohm_phi = 2.
		ohm_theta = 1.
		phi_0 = 0
		self.u0 = array([0,0,phi_0,0.,0,0,ohm_phi,ohm_theta,0,0])
		R = self.sys.radius
		m = self.sys.mass
		# consistent initial velocities
		vx = self.u0[4] = R*ohm_theta*np.cos(phi_0)
		vy = self.u0[5] = R*ohm_theta*np.sin(phi_0)
		# lagrange multipliers: used only used as a guess in RK methods
		self.u0[8], self.u0[9] = -m*ohm_phi*R*ohm_theta*np.sin(phi_0), m*ohm_phi*R*ohm_theta*np.cos(phi_0)
		self.s.initialize(self.u0,h=.01)
		self.s.time = 1.

	def test_run(self):
		self.s.run()
## 		self.s.plot(components=[6,7])
		npt.assert_array_almost_equal(self.s.final(), self.sys.exact(array([self.s.ts[-1]]),u0=self.u0)[:,0], decimal=1)


	experiences = {'50': (.01,50), '1000': (.1,1000.)}

	def run_experience(self, exp_name):
		self.s.shelf_name = 'bank_vr'
		h,time = self.experiences[exp_name]
		self.s.initialize(h=h,time=time)
		self.s.run()


class Test_VerticalRollingDisk_ML(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(McLachlan(), self.sys)

class Test_VerticalRollingDisk_H(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(NonHolonomicEnergy(), self.sys)

class Test_VerticalRollingDisk_Spark(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(Spark(2), self.sys)

# Test Spark on simple ODE

def minus_time(tx):
	return -tx[0]

class Test_SparkODE(object):
	def setUp(self):
		self.sys = ODESystem(minus_time)
		self.s = SingleStepSolver(Spark(4), self.sys)
		self.s.initialize(array([1.]))

	def test_run(self):
		self.s.run()
		exact = np.exp(-self.s.ats)
		print exact[-1]
		print self.s.final()
## 		npt.assert_array_almost_equal(self.s.aus, exact, 5)
		npt.assert_almost_equal(self.s.final(), exact[-1])
## 		plot(self.s.ats, np.vstack([self.s.aus, exact]).T)

# Test spark with Jay Example as a system

class Test_JayExample(object):
	def setUp(self):
		self.sys = JayExample()
## 		self.s.initialize(array([1.]))

	def test_spark(self):
		self.s = SingleStepSolver(Spark(2), self.sys)
		self.s.initialize(u0=array([1.,1.,1.]), time=1)
		self.s.run()
		print self.s.ts[-1]
		print self.s.final()
		exact = self.sys.exact(self.s.ts[-1],array([1.,1.,1.]))
		print exact
		npt.assert_array_almost_equal(self.s.final()[:2], exact[:2], decimal=2)

# RK DAE

def compare_exact(solver, u0, components, decimal=2):
	print solver.ts[-1]
	print solver.final()
	exact = solver.system.exact(solver.ts[-1], u0)
	npt.assert_array_almost_equal(solver.final()[:components], exact[:components], decimal=decimal)

def sq(x):
	return .5*x*x
def lin(x):
	return x
sq.der = lin

def test_rkdae():
	sys = GraphSystem(sq)
	u0 = array([0.,0.,1.])
	for s in range(1,4):
		sol = SingleStepSolver(RKDAE(RadauIIA.tableaux[s]), sys)
		sol.initialize(u0=u0, time=1)
		sol.run()
		yield compare_exact, sol, u0, 2

if __name__ == '__main__':
	from pylab import *
	t = Test_VerticalRollingDisk_H()
	t.setUp()
	t.run_experience('50')
