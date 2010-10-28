# -*- coding: UTF-8 -*-
from __future__ import division

import odelab

from odelab.scheme.stochastic import *
from odelab.system import *
from odelab.solver import *

def Vsin(self,t):
	return np.sin(t)

class Vlin(object):
	def __init__(self, t0, V0):
		self.t0 = t0
		self.V0 = V0
	
	def __call__(self,t):
		t0 = self.t0
		V0 = self.V0
		if t < t0:
			return 0.
		if t0 <= t < 2*t0:
			return V0*(t-t0)/t0
		if 2*t0 <= t < 3*t0:
			return V0
		if 3*t0 <= t < 4*t0:
			return -V0*(t-4*t0)/t0
		if 4*t0 <= t:
			return 0

		

class SimpleDiff(System):
	def __init__(self, V, gain=1e8):
		self.V = V
		self.gain = gain
		self.mass_mat = np.array([[1.,-self.gain],[0,0]])
		self.det_mat = np.array([[-self.gain, 0],[0,1.]])
	
	
	def deterministic(self,t,u):
		return np.dot(self.det_mat,u) - np.array([0,self.V(t)])
	
	def mass(self,t,u):
		return np.dot(self.mass_mat,u)
	
	def noise(self, t,u):
		return np.zeros_like(u)
	
	
class Test(object):
	def test_run(self):
		sys = SimpleDiff(V=Vlin(5.e-9,.01),gain=1e12)
		self.s = SingleStepSolver(EulerMaruyama(), sys)
		self.s.initialize(u0=np.array([0.,0]),h=2.5e-11,time=2.5e-8)
		self.s.run()
		

if __name__ == '__main__':
	t = Test()
	t.test_run()
	import pylab as pl
	pl.clf()
	t.s.plot()