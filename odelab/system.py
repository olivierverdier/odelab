# -*- coding: UTF-8 -*-
"""
:mod:`System` -- Systems
============================

Examples of ODE systems.

.. module :: system
	:synopsis: Examples of ODE systems.
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

"""
from __future__ import division

import numpy as np
from numpy import array

import scipy.sparse as sparse

import odelab.scheme.rungekutta as rk

class System(object):
	def __init__(self, f=None):
		if f is not None:
			self.f = f

	def label(self, component):
		return '%d' % component
	
	def preprocess(self, u0):
		return u0
	
	def postprocess(self, u1):
		return u1


class JayExample(System):
	r"""
	 The example in [jay06]_ §5. This is a test to check implementation of the
	 SRK-DAE2 methods given in [jay06]_. We want to compare our results to 
	 [jay06]_ Fig. 1.
	 
	 The exact solution to this problem is known as is
	
.. math::
	 
	y1(t) = \ee^t\\
	y2(t) = \ee^{-2t}\\
	z1(t) = \ee^{2t}
	 
We will compute the global error at :math:`t=1` at plot this relative to the
stepsize :math:`h`. This is what is done in [jay06]_ Fig.1.
	 
	 
:References:
	 
.. [jay06] Jay - Specialized Runge-Kutta Methods for index 2 DAEs (2006)
	"""

 
	def multi_dynamics(self, tu):
		y1,y2,z,t = tu
		return {
			rk.LobattoIIIA: array([y2 - 2*y1**2*y2, -y1**2]),
			rk.LobattoIIIB: array([y1*y2**2*z**2, np.exp(-t)*z - y1]),
			rk.LobattoIIIC: array([-y2**2*z, -3*y2**2*z]),
			rk.LobattoIIICs: array([2*y1*y2**2 - 2*np.exp(-2*t)*y1*y2, z]),
			rk.LobattoIIID: array([2*y2**2*z**2, y1**2*y2**2])
			}
	
	def constraint(self, tu):
		return array([tu[0]**2*tu[1] -  1])
	
	def state(self, u):
		return u[0:2]
	
	def lag(self, u):
		return u[2:3]
	
	def exact(self, t):
		return array([np.exp(t), np.exp(-2*t), np.exp(2*t)])
	
	def label(self, component):
		return ['y1','y2','z'][component]
	
	def test_exact(self, t):
		dyn = self.multi_dynamics(np.hstack([self.exact(t),t]))
		res = sum(v for v in dyn.values()) - array([np.exp(t), -2*np.exp(-2*t)])
		return res

class GraphSystem(System):
	r"""
	Trivial semi-explicit index 2 DAE of the form:
	
.. math::

		x' = 1\\
		y' = λ\\
		y  = f(x)
	"""
	
	def state(self, u):
		return u[:2]
	
	def lag(self, u):
		return u[-1:]
	
	def multi_dynamics(self, t, u):
		x,y = self.state(u)
		return {
			rk.LobattoIIIA: np.zeros_like(self.state(u)), 
			rk.LobattoIIIB: array([np.ones_like(x), self.lag(u)[0]]),
			}
	
	def constraint(self, t, u):
		x,y = self.state(u)
		return array([y - self.f(x)])
	
	def hidden_error(self, t, u):
		return self.lag(u)[0] - self.f.der(t,self.state(u)[0])

class ODESystem(System):
	"""
	Simple wrapper to transform an ODE into a semi-explicit DAE.
	"""
	def __init__(self, f, RK_class=rk.LobattoIIIA):
		self.f = f
		self.RK_class = RK_class
	
	def state(self, u):
		return u[:1]
	
	def lag(self,u):
		return u[1:] # should be empty
	
	def multi_dynamics(self, tu):
		return {self.RK_class: array([self.f(tu)])}
	
	def constraint(self, tu):
		return np.zeros([0,np.shape(tu)[1]])
	
	
def tensordiag(T):
	if len(np.shape(T)) == 3: # vector case
		assert T.shape[1] == T.shape[2]
		T = np.column_stack([T[:,s,s] for s in range(np.shape(T)[1])])
	return T

class ContactOscillator(System):
	"""
Example 5.2 in [MP]_.

This is the example presented in [MP]_ § 5.2. It is a nonlinear
perturbation of the contact oscillator.


.. [MP] R. McLachlan and M. Perlmutter, *Integrators for Nonholonomic Mechanical Systems*, J. Nonlinear Sci., **16**, 283-328., (2006)
	"""
	 

	def __init__(self, epsilon=0.):
		self.epsilon = epsilon
	
	def position(self, u):
		return u[:3]
	
	def velocity(self, u):
		return u[3:6]
	
	def state(self,u):
		return u[:6]
	
	def lag(self, u):
		return u[6:7]
		
	def assemble(self, q,v,l):
		return np.hstack([q,v,l])
	
	def vel_lag_split(self, vl):
		return vl[:3], vl[3:4]
	
	def vel_lag_stack(self, v,l):
		return np.hstack([v,l])
	
	def force(self, u):
		q = self.position(u) # copy?
		return -q - self.epsilon*q[2]*q[0]*array([q[2],np.zeros_like(q[0]),q[0]])
	
	def codistribution(self, u):
		q = self.position(u)
		return np.array([[np.ones_like(q[1]), np.zeros_like(q[1]), q[1]]])
	
	def energy(self, u):
		vel = self.velocity(u)
		q = self.position(u)
		return .5*(vel[0]**2 + vel[1]**2 + vel[2]**2 + q[0]**2 + q[1]**2 + q[2]**2 + self.epsilon*q[0]**2*q[2]**2)
	
	def initial(self, z0, e0=1.5, z0dot=0.):
		q0 = array([np.sqrt( (2*e0 - 2*z0dot**2 - z0**2 - 1) / (1 + self.epsilon*z0**22) ), 1., z0])
		p0 = array([-z0dot, 0, z0dot])
		v0 = p0
		l0 = ( q0[0] + q0[1]*q0[2] - p0[1]*p0[2] + self.epsilon*(q0[0]*q0[2]**2 + q0[0]**2*q0[1]*q0[2] ) ) / ( 1 + q0[1]**2 )
		return np.hstack([q0, v0, l0])
	
	def time_step(self, N=40):
		return 2*np.sin(np.pi/N)
	
	def reaction_force(self, u):
		reaction_force = np.tensordot(self.codistribution(u), self.lag(u), [0,0]) # a 3xsxs tensor
		reaction_force = tensordiag(reaction_force)
		return reaction_force
	
	def multi_dynamics(self, u):
		v = self.velocity(u)
		return {
		rk.LobattoIIIA: np.concatenate([v, np.zeros_like(v)]),
		rk.LobattoIIIB: np.concatenate([np.zeros_like(v), self.force(u) + self.reaction_force(u)])
		}
	
	def constraint(self, u):
		constraint = np.tensordot(self.codistribution(u), self.velocity(u), [1,0])
		constraint = tensordiag(constraint)
		return constraint

class Exponential(System):
	def __init__(self, nonlin, L):
		self.L = L
		self.nonlin = nonlin
	
	def linear(self):
		return self.L
	
	def f(self, t, u):
		return np.dot(self.linear(), u) + self.nonlin(t,u)

def zero_dynamics(t,u):
	return np.zeros_like(u)

class Linear(Exponential):
	def __init__(self, L):
		super(Linear, self).__init__(zero_dynamics, L)

class NoLinear(Exponential):
	def __init__(self, f, size):
		super(NoLinear,self).__init__(f, np.zeros([size,size]))
	

class Burgers(Exponential):
	def __init__(self, viscosity=0.03, size=128):
		self.viscosity = viscosity
		self.size = size
		self.initialize()
	
	def linear(self):
		return -self.laplace

		
class BurgersComplex(Burgers):
	def initialize(self):
		self.k = 2*np.pi*np.array(np.fft.fftfreq(self.size, 1/self.size),dtype='complex')
		self.k[len(self.k)/2] = 0
		self.laplace = self.viscosity*np.diag(self.k**2)
		x = np.linspace(0,1,self.size,endpoint=False)
		self.points = x - x.mean() # too lazy to compute x.mean manually here...
	
	def preprocess(self, u0):
		return np.fft.fft(u0)
	
	def postprocess(self, u1):
		return np.real(np.fft.ifft(u1))
	
	def nonlin(self, t, u):
		return -0.5j * self.k * np.fft.fft(np.real(np.fft.ifft(u)) ** 2)
	
