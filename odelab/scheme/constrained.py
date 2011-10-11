## -*- coding: UTF-8 -* -*- coding: UTF-8 -*-

from __future__ import division

import numpy as np

from odelab.scheme import Scheme


import odelab.newton as _rt

class McLachlan(Scheme):
	ur"""
Solver for the Lagrange-d'Alembert (LDA) equations using the
algorithm given by equation (4.18) in [MLPe06]_.

The scheme will work for any system as long as the necessary methods are implemented.
The adapted scheme reads as follows:

.. math::
	q_{1/2} &= q_0 + \frac{h}{2}H_p(q_0,p_0) \\
	p_1 &= p_0 + h\frac{H_q(q_{1/2},p_0) + H_q(q_{1/2},p_1)}{2} + h ∑_i λ_i α^i(q_{1/2})\\
	q_1 &= q_{1/2} + \frac{h}{2}H_p(q_1,p_1)\\
	&\langle{θ(q_1)},{H_p(q_1,p_1)}\rangle = 0



The :class:`odelab.system.System` object must implement:

* :meth:`odelab.system.System.velocity`
* :meth:`odelab.system.System.momentum`
* :meth:`odelab.system.System.position`
* :meth:`odelab.system.System.lag`
* :meth:`odelab.system.System.force`
* :meth:`odelab.system.System.codistribution`


:References:

.. [MLPe06] \R. McLachlan and M. Perlmutter, *Integrators for Nonholonomic Mechanical Systems*, J. Nonlinear Sci., **16** 283-328, (2006) :doi:`/10.1007/s00332-005-0698-1`
	"""

	def get_residual(self, t, u0, h):
		v0 = self.system.velocity(u0)
		momentum = self.system.momentum
		p0 = momentum(u0)
		qh = self.system.position(u0) + .5*h*v0
		qhp0 = np.hstack([qh,p0])
		codistribution = self.system.codistribution
		codistribution_h = codistribution(qh)
		def residual(du):
			u1 = u0+du
			q1 = self.system.position(u1)
			v1 = self.system.velocity(u1)
			l = self.system.lag(u1)
			p1 = momentum(u1)
			qhp1 = np.hstack([qh,p1])
			force = (self.system.force(qhp0) + self.system.force(qhp1))/2
			return np.hstack([
				q1 - qh - .5*h*v1,
				p1 - p0 - h * (force + np.dot(codistribution_h.T, l)),
				np.dot(codistribution(q1), v1),
				])
		return residual


class NonHolonomicEnergy(Scheme):

	def get_residual(self, t, u0, h):
		v0 = self.system.velocity(u0)
		q0 = self.system.position(u0)
		codistribution = self.system.codistribution
		l0 = self.system.lag(u0)
		def vector(du):
			u1 = u0 + du
			l1 = self.system.lag(u1)
			cod = codistribution(self.codistribution_q(u0,u1,h))
			return np.hstack([
				h*self.system.average_velocity(u0,u1),
				h*(self.system.average_force(u0,u1) + np.dot(cod.T,l1)),
				l1-l0 + h*np.dot(cod, self.system.average_velocity(u0,u1))])
		def residual(du):
			return du - vector(du)
		return residual

	def codistribution_q(self, u0, u1, h):
		return (self.system.position(u0)+self.system.position(u1))/2

class NonHolonomicEnergyEMP(NonHolonomicEnergy):
	"""
	Non Holonomic Energy preserving scheme with codistribution at the "explicit mid-point" q0 + h/2 v0
	"""
	def codistribution_q(self, u0, u1, h):
		return self.system.position(u0) + h/2*self.system.velocity(u0)

class NonHolonomicEnergy0(NonHolonomicEnergy):
	"""
	Non Holonomic Energy preserving scheme with codistribution at the starting point q0
	"""
	def codistribution_q(self, u0, u1, h):
		return self.system.position(u0)

class SymplecticEuler(Scheme):
	"""
Nonholonomic Symplectic Euler.
	"""
	root_solver = _rt.Newton

	def get_residual(self, t, u0, h):
		v0 = self.system.velocity(u0)
		q0 = self.system.position(u0)
		q1 = q0 + h*v0
		p0 = self.system.momentum(u0)
		force = self.system.force(q1)
		l0 = self.system.lag(u0)
		vl0 = np.hstack([v0,l0])
		codistribution_mid = self.system.codistribution((q0+q1)/2)
		def residual(dvl):
			u1 = np.hstack([q1, vl0+dvl]) # assuming that the system stores position,velocity
			v1 = self.system.velocity(u1)
			p1 = self.system.momentum(u1)
			l = self.system.lag(u1)
			codistribution = self.system.codistribution(q1)
			return np.hstack([p1 - p0 - h*(force + np.dot(codistribution_mid.T, l)), np.dot(codistribution, v1)])
		return residual

	def get_guess(self,t,u0,h):
		return np.zeros_like(np.hstack([self.system.velocity(u0),self.system.lag(u0)]))

	def reconstruct(self,root,t,u0,h):
		v0 = self.system.velocity(u0)
		dvl = root
		du = np.hstack([h*v0,dvl])
		return h, du

class NonHolonomicLeapFrog(Scheme):
	ur"""
Non-holonomic Leap Frog:

.. math::
	q_{i+1} &= q_i + h v_{i+1}\\
	v_{i+1} &= v_i + h (F(q_i) + A(q_i)^{T} λ_i)\\
	&A(q_i)(v_i+v_{i+1}) = 0

	"""
	root_solver = _rt.Newton

	def step(self, t, u0, h):
		q0 = self.system.position(u0)
		v0 = self.system.velocity(u0)
		force = self.system.force(q0)
		cod = self.system.codistribution(q0)
		def residual(vl1):
			u01 = np.hstack([q0,vl1])
			v1 = self.system.velocity(u01)
			l1 = self.system.lag(u01)
			return np.hstack([v1 - v0 - h*(force + np.dot(cod.T, l1)), np.dot(cod, v0+v1)])
		N = self.root_solver(residual)
		l0 = self.system.lag(u0)
		vl1 = N.run(np.hstack([v0,l0]))
		u01 = np.hstack([q0,vl1])
		v1 = self.system.velocity(u01)
		q1 = q0 + h*v1
		u1 = np.hstack([q1,vl1])
		return t+h, u1

