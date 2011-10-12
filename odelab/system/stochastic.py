#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.system.base import System
import numpy as np

class OrnsteinUhlenbeck(System):
	def __init__(self, sigma=1, theta=1, mu=1):
		self.sigma = sigma
		self.theta = theta
		self.mu = mu

	def mass(self, t, u):
		return u

	def deterministic(self, t, u):
		return self.theta*(self.mu - u)

	def noise(self, t, u):
		return self.sigma*np.identity(len(u))

class Signal(object):
	pass

class LinBumpSignal(Signal):
	def __init__(self, V0, t0):
		self.t0 = t0
		self.V0 = V0

	def __call__(self, t):
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

class SineSignal(Signal):
	def __init__(self, V0):
		self.V0 = V0

	def __call__(self, t):
		return self.V0*np.sin(2*np.pi*t/2e-8)


class Differentiator(System):
	kT = 300*1e-23

	def __init__(self, signal, C=1e-12,G=1e-4,A=300.):
		self.signal = signal
		self.C = C
		self.G = G
		self.A = A
		self.mass_mat = np.zeros([5,5])
		self.mass_mat[:2,:2] = np.array([[C,-C],[-C,C]])
		self.det_mat = np.zeros([5,5])
		self.det_mat[1:3,1:3] = np.array([[G,-G],[-G,G]])
		self.det_mat[0,3] = 1.
		self.det_mat[2,-1] = -1
		self.det_mat[3,1:3] = np.array([A,1])
		self.det_mat[-1,0] = 1.
		self.det_mat *= -1

	def mass(self,t,u):
		return np.dot(self.mass_mat,u)


	def V(self,t):
		V = np.zeros(5)
		V[-1] = self.signal(t)
		return V


	def deterministic(self, t, u):
		return np.dot(self.det_mat, u) + self.V(t)

	def noise(self, t, u):
		return np.sqrt(4*self.kT*self.G)*np.array([0,1.,-1,0,0]).reshape(-1,1)

