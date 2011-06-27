# -*- coding: UTF-8 -*-
from __future__ import division

from odelab.scheme import *
from odelab.scheme.constrained import *

from odelab.system import *
from odelab.solver import *

import numpy.testing as npt
import pylab as pl
from pylab import *

from nose.plugins.skip import SkipTest
import nose.tools as nt

SingleStepSolver.catch_runtime = False
SingleStepSolver.auto_save = False
SingleStepSolver.shelf_name = 'bank_constrained'

# Contact oscillator

class Harness_Osc(object):
	epsilon = 0.3
	def setUp(self):
		self.sys = ContactOscillator(epsilon=self.epsilon)
		self.set_solver()
		#self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
		self.s.time = 10.



	z0s = np.linspace(-.9,.9,10)*np.sqrt(2)
	N = 40

	def test_z0(self, i=5, nb_Poincare_iterations=10):
		z0 = self.z0s[i]
		h = self.sys.time_step(self.N)
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial_sin(z0), h=h, time=time)
		self.s.run()
		#self.s.plot(['radius'])
		npt.assert_almost_equal(self.sys.energy(self.s.final()), self.sys.energy(self.s.initial()), decimal=1)

	def run_chaotic(self, nb_Poincare_iterations=10):
		z0 = -.55
		h = self.sys.time_step(self.N)
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial_cos(z0), h=h, time=time)
		self.s.run()


class Test_McOsc(Harness_Osc):
	label = 'ML'
	def set_solver(self):
		self.s = SingleStepSolver(McLachlan(), self.sys)

class Test_JayOsc(Harness_Osc):
	N=5 # bigger time step to make test faster
	def set_solver(self):
		self.s = SingleStepSolver(Spark(3), self.sys)

class Test_HOsc(Harness_Osc):
	label = 'H'
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
		npt.assert_array_almost_equal(self.s.final()[:-1], self.sys.exact(array([self.s.final_time()]),u0=self.u0)[:,0], decimal=1)


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

class Test_VerticalRollingDisk_HM(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(NonHolonomicEnergyEMP(), self.sys)

class Test_VerticalRollingDisk_Spark(Harness_VerticalRollingDisk):
	def setup_solver(self):
		self.s = SingleStepSolver(Spark(2), self.sys)

class HarnessRobot(object):
	def setUp(self):
		s = SingleStepSolver(self.scheme, Robot())
		u0 = np.zeros(10)
		u0[4] = 1.
		u0[7] = 1.
		s.initialize(h=.2, time=1, u0 = u0)
		self.s = s

	def test_run(self):
		self.s.run()

	def test_energy(self):
		s = self.s
		s.run(time=100)
		nt.assert_almost_equal(s.system.energy(s.final()), s.system.energy(s.initial()), places=4)

class Test_Robot_ML(HarnessRobot):
	scheme = McLachlan()

	def test_energy(self):
		pass

class Test_Robot_H(HarnessRobot):
	scheme = NonHolonomicEnergy()

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
		exact = np.exp(-self.s.get_times())
		print exact[-1]
		print self.s.final()
## 		npt.assert_array_almost_equal(self.s.aus, exact, 5)
		npt.assert_almost_equal(self.s.final()[0], exact[-1])
## 		plot(self.s.ats, np.vstack([self.s.aus, exact]).T)

# Test spark with Jay Example as a system

class Test_JayExample(object):
	def setUp(self):
		self.sys = JayExample()
## 		self.s.initialize(array([1.]))

	def test_spark(self):
		self.s = SingleStepSolver(Spark(2), self.sys)
		self.s.initialize(u0=array([1.,1.,1.]), time=1, h=.05)
		self.s.run()
		print self.s.final_time()
		print self.s.final()
		exact = self.sys.exact(self.s.final_time(),array([1.,1.,1.]))
		print exact
		npt.assert_array_almost_equal(self.s.final()[:2], exact[:2], decimal=2)

def test_pendulum_ML():
	s = SingleStepSolver(McLachlan(), CirclePendulum())
	s.initialize(np.array([1.,0,0,0,0]))
	s.run()

def test_pendulum_NHE():
	s = SingleStepSolver(NonHolonomicEnergy(), SinePendulum())
	s.initialize(np.array([1.,0,0,0,0]))
	s.run()

class Harness_chaoticosc(object):
	def setUp(self):
		s = SingleStepSolver(self.scheme, ChaoticOscillator(3))
		u0 = np.zeros(15)
		N = 10 - 1
		n = 2 # j in range(N+1)
		angle = n*np.pi/2/N
		u0[:7] = np.array([np.cos(angle),.6,.4,.2,1.,1.,1.])
		u0[7:9] = np.array([0., np.sin(angle)])
		s.initialize(u0=u0, h=.05, time=1)
		self.s = s

	def test_run(self):
		s = self.s
		s.run()
		print s.system.energy(s.final())

	def test_energy(self):
		self.s.run()
		H1 = self.s.system.energy(self.s.final())
		H0 = self.s.system.energy(self.s.initial())
		print H0
		print H1
		npt.assert_almost_equal(H1, H0, decimal=self.energy_tol)

class Test_chaotic_ML(Harness_chaoticosc):
	scheme = McLachlan()
	energy_tol = 2

class Test_chaotic_H(Harness_chaoticosc):
	scheme = NonHolonomicEnergy()
	energy_tol = 10

class Harness_Chaplygin(object):
	def setUp(self):
		self.s = SingleStepSolver(self.solver_class(), Chaplygin(g=.1))
		#u0 = np.array([1.,0,.2,0,0,0,0])
		#u0 = np.array([1.,0,.8*np.pi/2,0,0,0,0])
		u0_Hilsen = np.array([1.,0,0,0,0,0,0])
		h_Hilsen = 1./300
		time_Hilsen = 1.
		u0_Jay = np.array([0,0,0,0,0,1.,0])
		h_Jay = .1
		time_Jay = 100
		self.s.initialize(u0=u0_Jay,time=time_Jay,h=h_Jay)
		#print self.s.system.energy(s.final())
		#nt.assert_equal(s.system.energy(self.s.events_array).shape, (len(self.s),))
		#return self.s

	def test_run(self):
		self.s.run()
		#nt.assert_almost_equal(s.system.energy(s.initial()), s.system.energy(s.final()))

class Test_Chaplygin_ML(Harness_Chaplygin):
	solver_class = McLachlan

class Test_Chaplygin_H(Harness_Chaplygin):
	solver_class = NonHolonomicEnergy






# RK DAE

class CompareExact(object):
	def __init__(self, name):
		self.description = name
	def __call__(self, solver, u0, components, decimal=2):
		solver.run()
		print solver.final_time()
		print solver.final()
		exact = solver.system.exact(solver.final_time(), u0)
		#npt.assert_array_almost_equal(solver.final()[:components], exact[:components], decimal=decimal)

def sq(x):
	return .5*x*x
def lin(x):
	return x
sq.der = lin


def test_rkdae():
	sys = GraphSystem(sq)
	u0 = array([0.,0.,1.])
	for s in range(2,4):
		sol = SingleStepSolver(RKDAE(RadauIIA.tableaux[s]), sys)
		sol.initialize(u0=u0, time=1)
		yield CompareExact('RadauIIA-{}'.format(s)), sol, u0, 2




