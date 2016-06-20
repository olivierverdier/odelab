from __future__ import division

import numpy as np

from . import NonHolonomic

class Robot(NonHolonomic):
	def position(self,u):
		return u[:4]

	def velocity(self, u):
		return u[4:8]

	def lag(self,u):
		return u[8:10]

	def codistribution(self, u):
		q2 = self.position(u)[2]
		cod = np.zeros([2,4])
		cod[0,0] = cod[1,1] = 1.
		cod[0,3] = -np.cos(q2)
		cod[1,3] = -np.sin(q2)
		return cod

	def force(self, u):
		q = self.position(u)
		f = np.zeros_like(q)
		f[3] = -10*np.cos(q[3])
		return f

	def average_force(self, u0, u1):
		q0 = self.position(u0)
		t0 = q0[3]
		t1 = self.position(u1)[3]
		if np.allclose(t0,t1):
			f = -10*np.cos(t0)
		else:
			f = -10*(np.sin(t1) - np.sin(t0))/(t1-t0) # check this!
		fvec = np.zeros_like(q0)
		fvec[3] = f
		return fvec

	def energy(self, u):
		q3 = self.position(u)[3]
		return .5*np.sum(self.momentum(u) * self.velocity(u), axis=0) + 10*np.sin(q3)

class VerticalRollingDisk(NonHolonomic):
	"""
	Vertical Rolling Disk
	"""

	size = 10 # 4+4+2

	def __init__(self, mass=1., radius=1., Iflip=1., Irot=1.):
		"""
		:mass: mass of the disk
		:radius: Radius of the disk
		:Iflip: inertia momentum around the "flip" axis
		:Irot: inertia momentum, around the axis of rotation symmetry of the disk (perpendicular to it)
		"""
		self.mass = mass
		self.radius = radius
		self.Iflip = Iflip
		self.Irot = Irot

	#def label(self, component):
		#return ['x','y',u'φ',u'θ','vx','vy',u'ωφ',u'ωη',u'λ1',u'λ2'][component]

	def position(self, u):
		"""
		Positions x,y,φ (SE(2) coordinates), θ (rotation)
		"""
		return u[:4]

	def velocity(self, u):
		return u[4:8]

	def average_force(self, u0, u1):
		return self.force(u0) # using the fact that the force is zero in this model

	def lag(self,u):
		return u[8:10]

	def codistribution(self, u):
		q = self.position(u)
		phi = q[2]
		R = self.radius
		one = np.ones_like(phi)
		zero = np.zeros_like(phi)
		return np.array([[one, zero, zero, -R*np.cos(phi)],[zero, one, zero, -R*np.sin(phi)]])

	def state(self,u):
		return u[:8]

	def force(self,u):
		return np.zeros_like(self.position(u))

	def qnorm(self, ut):
		return np.sqrt(ut[0]**2 + ut[1]**2)

	def energy(self, ut):
		return .5*(self.mass*(ut[4]**2 + ut[5]**2) + self.Iflip*ut[6]**2 + self.Irot*ut[7]**2)

	def exact(self,t,u0):
		"""
		Exact solution for initial condition u0 at times t

:param array(N) t: time points of size N
:param array(8+) u0: initial condition
:return: a 10xN matrix of the exact solution
		"""
		ohm_phi,ohm_theta = u0[6:8]
		R = self.radius
		rho = ohm_theta*R/ohm_phi
		x_0,y_0,phi_0,theta_0 = u0[:4]
		phi = ohm_phi*t+phi_0
		one = np.ones_like(t)
		m = self.mass
		return np.vstack([rho*(np.sin(phi)-np.sin(phi_0)) + x_0,
					-rho*(np.cos(phi)-np.cos(phi_0)) + y_0,
					ohm_phi*t+phi_0,
					ohm_theta*t+theta_0,
					R*np.cos(phi)*ohm_theta,
					R*np.sin(phi)*ohm_theta,
					ohm_phi*one,
					ohm_theta*one,
					-m*ohm_phi*R*ohm_theta*np.sin(phi),
					m*ohm_phi*R*ohm_theta*np.cos(phi),])

	def initial(self, u00):
		"""
		Make sure that the constraints are fulfilled at the initial conditions.
		"""
		u0 = np.copy(u00)
		phi = u0[2]
		vtheta = u0[7]
		u0[4] = np.cos(phi)*vtheta
		u0[5] = np.sin(phi)*vtheta
		return u0
