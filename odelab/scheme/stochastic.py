# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np
import scipy.linalg as sl

from odelab.scheme import Scheme
from odelab.newton import FSolve, Newton

class EulerMaruyama(Scheme):
	def step(self,t,u):
		h = self.h
		system = self.system
		noise = np.random.normal(size=[len(system.noise(t,u).T)])
		def residual(v):
			return (system.mass(t+h,v) - system.mass(t,u))/h - system.deterministic(t+h,v) - np.sqrt(h)/h*np.dot(system.noise(t,u),noise)
		N = Newton(residual)
## 		N.xtol = min(N.xtol, h*1e-4)
		result = N.run(u)
		return t+h, result
	
	def linstep(self,t,u):
		return t+self.h, sl.solve(self.system.mass_mat-self.h*self.system.det_mat, np.dot(self.system.mass_mat,u)-self.h*self.system.V(t+self.h))
	
