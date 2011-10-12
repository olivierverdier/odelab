#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

from odelab.system.base import *

def tensordiag(T):
	if len(np.shape(T)) == 3: # vector case
		assert T.shape[1] == T.shape[2]
		T = np.column_stack([T[:,s,s] for s in range(np.shape(T)[1])])
	return T

class NonHolonomic(MechanicalSystem):
	"""
	Creates a DAE system out of a non-holonomic one, suitable to be used with the :class:`odelab.scheme.constrained.Spark` scheme.
	"""
	def constraint(self, u):
		constraint = np.tensordot(self.codistribution(u), self.velocity(u), [1,0])
		constraint = tensordiag(constraint)
		return constraint

	def reaction_force(self, u):
		# a nxsxs tensor, where n is the degrees of freedom of the position:
		reaction_force = np.tensordot(self.codistribution(u), self.lag(u), [0,0])
		reaction_force = tensordiag(reaction_force)
		return reaction_force

	def multi_dynamics(self, ut):
		"""
		Split the Lagrange-d'Alembert equations in a Spark LobattoIIIA-B method.
		It splits the vector field into two parts.
		f_1 = [v, 0]
		f_2 = [0, f + F]

		where f is the external force and F is the reaction force stemming from the constraints.
		"""
		v = self.velocity(ut)
		return {
		rk.LobattoIIIA: np.concatenate([v, np.zeros_like(v)]),
		rk.LobattoIIIB: np.concatenate([np.zeros_like(v), self.force(ut) + self.reaction_force(ut)])
		}

	def momentum(self, u):
		"""
		Default is to have an identity mass matrix.
		"""
		return self.velocity(u)

	def average_velocity(self, u0, u1):
		"""
		Default value for the average velocity.
		"""
		return (self.velocity(u0) + self.velocity(u1))/2

	def assemble(self, q,v,l):
		"""
		Default value for the assemble method.
		"""
		return np.hstack([q,v,l])






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

	def label(self, component):
		return ['x','y',u'φ',u'θ','vx','vy',u'ωφ',u'ωη',u'λ1',u'λ2'][component]

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

class Pendulum(NonHolonomic):
	def __init__(self, constraint=None, codistribution=None):
		self._constraint = constraint
		self._codistribution = codistribution

	def position(self, u):
		return u[:2]

	def velocity(self, u):
		return u[2:4]

	def lag(self,u):
		return u[4:5]

	def codistribution(self, u):
		return self._codistribution(u)

	def force(self, u):
		return -np.array([0,1])

	def average_force(self,u0,u1):
		return self.force(u0)

	def constraint(self, u):
		return self._constraint(u)

class CirclePendulum(Pendulum):
	def codistribution(self, u):
		return self.position(u).reshape(1,-1)

	def constraint(self,u):
		x,y =  u[0], u[1]
		return np.sqrt(x*x + y*y)

class SinePendulum(Pendulum):
	def codistribution(self, u):
		x,y = self.position(u)
		return np.hstack([np.cos(x), -np.ones_like(y)]).reshape(1,-1)

	def constraint(self, u):
		x,y = self.position(u)
		return y - np.sin(x)

def average_x_sqsq_force(x0,x1,y0,y1):
	"""
	mean force on the x component for the potential :math:`-x^2y^2`
	"""
	return (x0*y0**2 + x1*y1**2)/4 + (2*y0*y1*(x0+x1) + x0*y1**2 + x1*y0**2)/12

def add_average_sqsq_component(f,q0,q1,i,j):
	"""
	mean force for a potential :math:`- q_i^2 q_j^2`
	"""
	f[i] += average_x_sqsq_force(q0[i],q1[i],q0[j],q1[j])
	f[j] += average_x_sqsq_force(q0[j],q1[j],q0[i],q1[i])

class ChaoticOscillator(NonHolonomic):
	def __init__(self, size=3):
		"""
:param size: the underlying size; the total dimension is 2n+1 + 2n+1 + 1 = 4n+3; that is 15 for the default size n=3
		"""
		self.size = size

	def position(self,u):
		return u[:2*self.size+1]

	def velocity(self, u):
		q = 2*self.size + 1
		return u[q:2*q]

	def lag(self, u):
		q = 2*self.size + 1
		return u[2*q:2*q+1]

	def force(self, u):
		q = self.position(u)
		x = q[0]
		n = self.size
		y = q[1:n+1]
		z = q[n+1:2*n+1]
		force = q.copy()
		force[n+1] += z[1]*z[1]*z[0]
		force[n+2] += z[0]*z[0]*z[1]
		force[1:n+1] += y*z*z
		force[n+1:2*n+1] += y*y*z
		return -force

	def average_force(self, u0, u1):
		q0 = self.position(u0)
		q1 = self.position(u1)
		n = self.size
		force = (q0 + q1)/2.
		add_average_sqsq_component(force,q0,q1,n+1,n+2)
		for i in range(n):
			add_average_sqsq_component(force,q0,q1,1+i,1+n+i)
		return -force


	def codistribution(self, u):
		q = self.position(u)
		n = self.size
		return np.hstack([1, np.zeros(n), q[n+1:2*n+1]]).reshape(1,-1)

	def energy(self, u):
		q = self.position(u)
		v = self.velocity(u)
		x = q[0]
		n = self.size
		y = q[1:n+1]
		z = q[n+1:2*n+1]
		return .5*(np.sum(v*v,axis=0) + np.sum(q*q,axis=0) + np.sum(y*y*z*z, axis=0) + z[0]**2*z[1]**2)

class Chaplygin(NonHolonomic):
	"""
Models the Chaplygin Sleigh. It has the Lagrangian

.. math::

	L(x,y,θ,\dot{x},\dot{y},\dot{θ}) = \dot{x}^2 + \dot{y}^2 + \dot{θ}^2 - gy

One encounters the following situations in the literature [Bloch]_, [Rheinboldt]_:

:Chaplygin sleigh: g = 0
:Knife edge, or skate on an inclined plane: a = 0

The nonholonomic constraint is given by

.. math::

	\cos(θ)\dot{x} - \sin(θ)\dot{y} - a \dot{θ}

.. [Bloch] Bloch: *Nonholonomic Mechanics and Control*
.. [Rheinboldt] Rabier, Rheinboldt: *Nonholonomic Motion of Rigid Mechanical Systems from a DAE Viewpoint*
	"""
	def __init__(self, g=1., length=1.):
		"""
g is the gravity. In principle it is between -1 and 1.
length is the distance between the contact point and the centre of gravity
		"""
		self.g = g
		self.length = length
		self.mass_matrix = np.identity(3)

	def position(self, u):
		return u[:3]

	def velocity(self, u):
		return u[3:6]

	def lag(self, u):
		return u[6:7]

	def momentum(self, u):
		return np.dot(self.mass_matrix, self.velocity(u))

	def force(self, u):
		f = np.zeros(3)
		f[1] = -self.g
		return f

	def codistribution(self, u):
		theta = self.position(u)[2]
		return np.hstack([-np.sin(theta), np.cos(theta), -self.length]).reshape([1,-1])

	def energy(self, u):
		y = self.position(u)[1]
		return .5*np.sum(self.momentum(u) * self.velocity(u), axis=0) + self.g*y

	def average_force(self, u0, u1):
		return self.force(u0)

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


# for compatibility:
from contactoscillator import *
