#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from . import NonHolonomic, np


class ContactSystem(NonHolonomic):
	"""
General class of 3D mechanical systems with the constraint codistribution

.. math::
	\dd x + y \dd z

The energy is assumed to be

.. math::
	H = v_x^2 + v_y^2 + v_z^2 + U(x,y,z)
	"""
	size = 7 # 3+3+1
	def label(self, component):
		return [u'x',u'y',u'z',u'ẋ',u'ẏ',u'ż',u'λ'][component]

	def position(self,u):
		return u[:3]

	def momentum(self,u):
		return u[3:6]

	def velocity(self,u):
		return self.momentum(u)

	def lag(self,u):
		return u[6:7]

	def codistribution(self, u):
		y = self.position(u)[1]
		return np.array([[np.ones_like(y), np.zeros_like(y), y]])

	def potential(self,u):
		"""
		Potential energy
		"""
		q = self.position(u)
		q2 = np.square(q)
		return (q2[0] + q2[1] + q2[2])/2

	def force(self, u):
		q = self.position(u)
		return -q

	def energy(self, u):
		vel = self.velocity(u)
		q = self.position(u)
		v2 = np.square(vel)
		q2 = np.square(q)
		return (v2[0] + v2[1] + v2[2])/2 + self.potential(u)

	def initial(self,u00):
		ur"""
		Figure out a valid initial condition by replacing :math:`v_x` by :math:`-yv_z`, and the true value of :math:`λ`.
		"""
		u0 = u00.copy()
		force = self.force(u0)
		Fx = force[0]
		Fz = force[2]
		y = self.position(u0)[1]
		vel = self.velocity(u0)
		u0[3] = - y * u0[5]
		u0[6] = -(Fx + y*Fz + vel[1]*vel[2])/(1+y**2)
		return u0


class ContactOscillator(ContactSystem):
	"""
Example 5.2 in [MP]_.

This is the example presented in [MP]_ § 5.2. It is a nonlinear
perturbation of the contact oscillator.


.. [MP] R. McLachlan and M. Perlmutter, *Integrators for Nonholonomic Mechanical Systems*, J. Nonlinear Sci., **16**, No. 4, 283-328., (2006). :doi:`10.1007/s00332-005-0698-1`
	"""

	def __init__(self, epsilon=0.):
		self.epsilon = epsilon

	def state(self,u):
		return u[:6]

	def potential(self,u):
		q = self.position(u)
		q2 = np.square(q)
		return super(ContactOscillator,self).potential(u) + self.epsilon*q2[0]*q2[2]/2

	def force(self, u):
		q = self.position(u)
		return -q - self.epsilon*q[2]*q[0]*np.array([q[2],np.zeros_like(q[0]),q[0]])

	def average_force(self, u0, u1):
		q0, q1 = self.position(u0), self.position(u1)
		x0,z0 = q0[0],q0[2]
		x1,z1 = q1[0],q1[2]
		qm = (q0+q1)/2
		px = (x0*z0**2 + x1*z1**2)/4 + (2*z0*z1*(x0+x1) + x0*z1**2 + x1*z0**2)/12
		pz = (z0*x0**2 + z1*x1**2)/4 + (2*x0*x1*(z0+z1) + z0*x1**2 + z1*x0**2)/12
		return -qm - self.epsilon*np.array([px, np.zeros_like(q0[0]), pz])

	def energy(self, u):
		vel = self.velocity(u)
		q = self.position(u)
		v2 = np.square(vel)
		q2 = np.square(q)
		return self.potential(u) + (v2[0] + v2[1] + v2[2] )/2

	def energy_y(self,u):
		"""
		Return Hy = (y**2 + vy**2)/2
		"""
		return (u[1]**2 + u[4]**2)/2

	def radius(self, u):
		"""
		Conserved quantity equal to H - Hy.
		"""
		q = self.position(u)
		vel = self.velocity(u)
		x2 = q[0]**2
		return .5*(x2 + (1+self.epsilon*x2)*q[2]**2 + (1+q[1]**2)*vel[2]**2)

	def initial_cos(self, z0, H0=1.5, Hy=.5, z0dot=0.):
		"""
		Initial condition assuming y(t) = √(2Hy)*cos(t)
		"""
		b = np.sqrt(2*Hy)
		q0 = np.array([np.sqrt( (2*H0 - 2*z0dot**2 - z0**2 - 1) / (1 + self.epsilon*z0**2) ), b, z0])
		p0 = np.array([-b*z0dot, 0, z0dot])
		v0 = p0
		l0 = ( q0[0] + q0[1]*q0[2] - p0[1]*p0[2] + self.epsilon*(q0[0]*q0[2]**2 + q0[0]**2*q0[1]*q0[2] ) ) / ( 1 + q0[1]**2 )
		return np.hstack([q0, v0, l0])

	def initial_sin(self,z0, H0=1.5, Hy=.5, z0dot=0.):
		"""
		Initial condition assuming y(t) = √(2Hy)*sin(t)
		"""
		x0 = np.sqrt((2*(H0-Hy) - z0**2 - z0dot**2)/(1 + self.epsilon*z0**2))
		q0 = np.array([x0, 0, z0])
		a = np.sqrt(2*Hy)
		p0 = np.array([0, a, z0dot])
		v0 = p0
		l0 = x0 - a*z0dot + self.epsilon*z0**2
		return np.hstack([q0, v0, l0])

	def time_step(self, N=40):
		return 2*np.sin(np.pi/N)

class NonReversibleContactOscillator(ContactOscillator):
	ur"""
Non Reversible contact oscillator. The new Hamiltonian is obtained from that of the Contact Oscillator by

.. math::

	H(q,p) = \frac{1}{2}\bigl(\|q\|^2 + \|p + \grad S\|^2\bigr)

The perturbation is :math:`\grad S`, where :math:`S` depends only on :math:`q`.
The new variables :math:`Q`, :math:`P` are obtained with the transformation:

.. math::

	Q &= q \\
	P &= p - \grad S

	"""
	def s(self,z):
		return np.cos(z)
	def sder(self,z):
		return -np.sin(z)
	def ssec(self,z):
		return -np.cos(z)

	perturbation_size = .2
	def perturbation(self, q):
		"""
Gradient of the generating function S = s(z).
		"""
		pert = np.zeros_like(q)
		z = q[2]
		pert[-1] = self.sder(z)
		return pert

	def velocity(self, u):
		return self.momentum(u) + self.perturbation(self.position(u))

	def momentum(self,u):
		return u[3:6]

	def force(self,u):
		orig_force = super(NonReversibleContactOscillator,self).force(u) # works because force depends only on position
		q = self.position(u)
		orig_force[2] -= self.ssec(q[2])*self.velocity(u)[2]
		return orig_force

	def average_force(self, u0,u1):
		orig_average_force = super(NonReversibleContactOscillator,self).average_force(u0,u1)
		p0 = self.momentum(u0)[2]
		p1 = self.momentum(u1)[2]
		z0 = self.position(u0)[2]
		z1 = self.position(u1)[2]
		if np.allclose(z0,z1):
			perturbation = self.ssec(z0)*((p0+p1)/2 + self.sder(z0))
		else:
			perturbation = (self.sder(z1)*(p1+self.sder(z1)/2) - self.sder(z0)*(p0+self.sder(z0)/2) - (self.s(z1)-self.s(z0))/(z1-z0)*(p1-p0) )/(z1-z0)
		return orig_average_force - np.array([0.,0.,perturbation])

	def average_velocity(self,u0,u1):
		average_momentum = (self.momentum(u0) + self.momentum(u1))/2
		z0 = self.position(u0)[2]
		z1 = self.position(u1)[2]
		if np.allclose(z0,z1):
			perturbation = self.sder(z0)
		else:
			perturbation = (self.s(z1) - self.s(z0))/(z1-z0)
		average_momentum[2] += perturbation
		return 	average_momentum


	def initial_cos(self, z0, H0=1.5, Hy=.5, z0dot=0.):
		"""
		Initial condition assuming y(t) = √(2Hy)*cos(t)
		"""
		u0 = super(NonReversibleContactOscillator,self).initial_cos(z0,H0,Hy,z0dot)
		u0[5] -= self.sder(u0[2])
		return u0

	def energy(self, u):
		p = self.momentum(u).copy()
		p[2] += self.sder(u[2])
		q = self.position(u)
		p2 = np.square(p)
		q2 = np.square(q)
		return (p2[0] + p2[1] + p2[2] + q2[0] + q2[1] + q2[2] + self.epsilon*q2[0]*q2[2])/2

