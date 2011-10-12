#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.system.base import *

class MechanicalSystem(System):
	def velocity(self, u):
		return self.momentum(u)

	def position(self, u):
		"""select the position component"""

	def lag(self, u):
		"""select the lagrangian component"""

	def vel_lag_split(self, vl):
		"""splits between velocity and lagrangian"""

	def vel_lag_stack(self,v,l):
		"""stacks together the velocity and lagrangian parts"""

	def force(self, u):
		r"""compute the force as a function of `u`
		It may be interpreted as :math:`-\frac{∂H}{∂q}`"""

	def codistribution(self, u):
		"""compute the codistribution matrix at `u`"""



class HarmonicOscillator(MechanicalSystem):
	def position(self,u):
		return u[:1]

	def momentum(self, u):
		return u[1:2]

	def force(self, u):
		q = self.position(u)
		return -q

	def energy(self, ut):
		q = self.position(ut)
		p = self.position(ut)
		return (np.square(q) + np.square(p))/2


class HenonHeiles(MechanicalSystem):
	def position(self,u):
		return u[:2]

	def momentum(self,u):
		return u[2:4]

	def force(self, u):
		q0,q1 = self.position(u)
		return -np.array([q0*(1+2*q1), q1 - q1**2 + q0**2])

	def energy(self, ut):
		q = self.position(ut)
		p2 = np.square(self.momentum(ut))
		q2 = np.square(q)
		return (p2[0]+p2[1] + q2[0]+q2[1])/2 + q2[0]**2*q[1] - q2[1]*q[1]/3

	def initial(self, H, y,py,x=0):
		px = np.sqrt(2*H - (py**2 + y**2 + x**2 + 2*(x**2*y - y**3/3)))
		return np.array([x,y,px,py])





