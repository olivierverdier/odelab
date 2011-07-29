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


# Contact oscillator

class Harness_Osc(object):
	epsilon = 0.3
	def setUp(self):
		self.sys = ContactOscillator(epsilon=self.epsilon)
		self.set_scheme()
		self.s = SingleStepSolver(self.scheme, self.sys)
		#self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
		self.s.time = 10.



	z0s = np.linspace(-.9,.9,10)*np.sqrt(2)
	N = 40

	def test_z0(self, i=5, nb_Poincare_iterations=10):
		z0 = self.z0s[i]
		h = self.sys.time_step(self.N)
		self.scheme.h = h
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial_sin(z0), time=time)
		self.s.run()
		#self.s.plot(['radius'])
		npt.assert_almost_equal(self.sys.energy(self.s.final()), self.sys.energy(self.s.initial()), decimal=1)
		with self.s.open_store() as events:
			energy = self.s.system.energy(events)

	def run_chaotic(self, nb_Poincare_iterations=10):
		z0 = -.55
		h = self.sys.time_step(self.N)
		self.scheme.h = h
		time = nb_Poincare_iterations*self.N*h
		self.s.initialize(u0=self.sys.initial_cos(z0), h=h, time=time)
		self.s.run()


class Test_McOsc(Harness_Osc):
	label = 'ML'
	def set_scheme(self):
		self.scheme = McLachlan()

class Test_JayOsc2(Harness_Osc):
	N=10 # bigger time step to make test faster
	def set_scheme(self):
		self.scheme = Spark(2)

class Test_JayOsc3(Harness_Osc):
	N=5 # bigger time step to make test faster
	def set_scheme(self):
		self.scheme = Spark(3)

class Test_HOsc(Harness_Osc):
	label = 'H'
	N=5 # bigger time step to make test faster
	def set_scheme(self):
		self.scheme = NonHolonomicEnergy()

class Test_NROsc(Test_McOsc):
	epsilon = 0.
	def setUp(self):
		self.sys = NonReversibleContactOscillator()
		self.set_scheme()
		self.s = SingleStepSolver(self.scheme, self.sys)
		self.s.time = 10.


# Vertical Rolling Disk

class Harness_VerticalRollingDisk(object):
	h = .01
	def setUp(self):
		self.sys = VerticalRollingDisk()
		self.setup_scheme()
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
		self.scheme.h = self.h
		self.s = SingleStepSolver(self.scheme, self.sys)
		self.s.initialize(self.u0,)
		self.s.time = 1.

	def test_run(self):
		self.s.run()
## 		self.s.plot(components=[6,7])
		self.check_solution()

	def check_solution(self, decimal=1):
		npt.assert_array_almost_equal(self.s.final()[:8], self.sys.exact(array([self.s.final_time()]),u0=self.u0)[:8,0], decimal=decimal)

	def check_energy(self, decimal):
		energy = self.sys.energy
		npt.assert_almost_equal(energy(self.s.final()), energy(self.s.initial()), decimal=decimal)

	experiences = {'50': (.01,50), '1000': (.1,1000.)}

	def run_experience(self, exp_name):
		h,time = self.experiences[exp_name]
		self.s.initialize(h=h,time=time)
		self.s.run()


class Test_VerticalRollingDisk_ML(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = McLachlan()

class Test_VerticalRollingDisk_H(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = NonHolonomicEnergy()

class Test_VerticalRollingDisk_H0(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = NonHolonomicEnergy0()

class Test_VerticalRollingDisk_HM(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = NonHolonomicEnergyEMP()

class Test_VerticalRollingDisk_Spark2(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = Spark(2)

class Test_VerticalRollingDisk_Spark3(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = Spark(3)

class Test_VerticalRollingDisk_Spark4(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = Spark(4)

class Test_VerticalRollingDisk_SE(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = SymplecticEuler()

class Test_VerticalRollingDisk_LF(Harness_VerticalRollingDisk):
	def setup_scheme(self):
		self.scheme = NonHolonomicLeapFrog()

class HarnessRobot(object):
	def setUp(self):
		s = SingleStepSolver(self.scheme, Robot())
		u0 = np.zeros(10)
		u0[4] = 1.
		u0[7] = 1.
		self.scheme.h = .2
		s.initialize(time=1, u0 = u0)
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
		scheme = Spark(4)
		scheme.h = .1
		self.s = SingleStepSolver(scheme, self.sys)
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
		scheme = Spark(2)
		scheme.h = .05
		self.s = SingleStepSolver(scheme, self.sys)
		self.s.initialize(u0=array([1.,1.,1.]), time=1,)
		self.s.run()
		print self.s.final_time()
		print self.s.final()
		exact = self.sys.exact(self.s.final_time(),array([1.,1.,1.]))
		print exact
		npt.assert_array_almost_equal(self.s.final()[:2], exact[:2], decimal=2)

def test_pendulum_ML():
	s = SingleStepSolver(McLachlan(h=.1), CirclePendulum())
	s.initialize(np.array([1.,0,0,0,0]))
	s.run()

def test_pendulum_NHE():
	s = SingleStepSolver(NonHolonomicEnergy(h=.1), SinePendulum())
	s.initialize(np.array([1.,0,0,0,0]))
	s.run()

class Harness_chaoticosc(object):
	def setUp(self):
		self.scheme.h = .05
		s = SingleStepSolver(self.scheme, ChaoticOscillator(3))
		u0 = np.zeros(15)
		N = 10 - 1
		n = 2 # j in range(N+1)
		angle = n*np.pi/2/N
		u0[:7] = np.array([np.cos(angle),.6,.4,.2,1.,1.,1.])
		u0[7:9] = np.array([0., np.sin(angle)])
		s.initialize(u0=u0,  time=1)
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
		self.s = SingleStepSolver(self.scheme, Chaplygin(g=.1))
		#u0 = np.array([1.,0,.2,0,0,0,0])
		#u0 = np.array([1.,0,.8*np.pi/2,0,0,0,0])
		u0_Hilsen = np.array([1.,0,0,0,0,0,0])
		h_Hilsen = 1./300
		time_Hilsen = 1.
		u0_Jay = np.array([0,0,0,0,0,1.,0])
		h_Jay = .1
		time_Jay = 100
		self.scheme.h = h_Jay
		self.s.initialize(u0=u0_Jay,time=time_Jay,)
		#print self.s.system.energy(s.final())
		#nt.assert_equal(s.system.energy(self.s.events_array).shape, (len(self.s),))
		#return self.s

	def test_run(self):
		self.s.run()
		#nt.assert_almost_equal(s.system.energy(s.initial()), s.system.energy(s.final()))

class Test_Chaplygin_ML(Harness_Chaplygin):
	scheme = McLachlan()

class Test_Chaplygin_H(Harness_Chaplygin):
	scheme = NonHolonomicEnergy()






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
		scheme = RKDAE(RadauIIA.tableaux[s])
		scheme.h = .1
		sol = SingleStepSolver(scheme, sys)
		sol.initialize(u0=u0, time=1)
		yield CompareExact('RadauIIA-{}'.format(s)), sol, u0, 2



if __name__ == '__main__':
	t = Test_NROsc()
	t.setUp()
	t.test_z0()
