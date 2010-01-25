# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np
from numpy import array, dot
from numpy.linalg import norm, inv, solve
import pylab as PL
from pylab import plot, legend


def jacobian(F,x,h=1e-6):
	"""
	Numerical Jacobian at x.
	"""
	if np.isscalar(x):
		L = 1
	else:
		L = len(x)
	vhs = h * np.identity(L)
	Fs = array([F(x+vh) for vh in vhs])
	grad = (Fs - F(x))/h
	return array(grad).T

class Newton(object):
	"""
	Simple Newton solver to solve F(x) = level.
	"""
	def __init__(self, F=None, level=0.):
		self.F = F
		self.level = level
	
	h = 1e-6
	def der(self, x):
		return jacobian(self.F,x,self.h)
	
	maxiter = 100
	tol = 1e-7
	def run(self, x0):
		"""
		Run the Newton iteration.
		"""
		if np.isscalar(x0):
			x = x0
		else:
			x = x0.copy()
		for i in xrange(self.maxiter):
			d = self.der(x)
			y = self.level - self.F(x)
			if np.isscalar(y):
				incr = y/d.item()
			else:
				incr = solve(d, y)
			x += incr
			if norm(incr) < self.tol:
				break
##			if self.is_zero(x):
##				break
		else:
			raise Exception("Newton algorithm did not converge. ∆x=%.2e" % incr)
		self.required_iter = i
		return x
	
	def is_zero(self, x):
		res = norm(self.F(x) - self.level)
		return res < self.tol


class ODESolver (object):

	def __init__(self, f=None):
		self.f = f


	def increment_stepsize(self):
		"""
		Change the step size based on error estimation.
		To be overridden for a variable step size method.
		"""
		pass

	max_iter = 1000000
	
	# default values for h and time
	h = .01
	time = 1.

	def initialize(self, u0=None, t0=0, h=None, time=None):
		"""
		Initialize the solver to the initial condition u(t0) = u0.
		"""
		if u0 is None:
			u0 = self.us[0]
			t0 = self.ts[0]
		self.ts = [t0]
		self.us = [u0]
		if h is not None:
			self.h = h
		if time is not None:
			self.time = time

	def generate(self, t, u, tf):
		"""
		Generates the (t,u) values until t > tf
		"""
		for i in xrange(self.max_iter):
			t, u = self.step(t, u)
			if t > tf:
				break
			yield t, u
			self.increment_stepsize()
		else:
			raise Exception("Final time not reached")
		
	
	def run(self, time=None):
		if time is None:
			time = self.time
		# start from the last time we stopped
		t = t0 = self.ts[-1]
		u = self.us[-1]
		qs = list(self.generate(t, u, t0 + time))
		
		self.ts.extend(q[0] for q in qs)
		self.us.extend(q[1] for q in qs)
		
		self.ats = array(self.ts)
		self.aus = array(self.us)


	def plot(self, components=None):
		"""
		Plot some components of the solution.
		"""
		if components is None:
			components = range(len(self.us[0]))
		for component in components:
			plot(self.ats, self.aus[:,component], ',-', label=self.labels[component])
		PL.xlabel('time')
		PL.legend()

	def plot2D(self):
		"""
		Plot ux vs uy
		"""
		plot(self.aus[:,0],self.aus[:,1], '.-')
		xlabel('ux')
		ylabel('uy')
	
	quiver_res = 20
	def quiver(self):
		mins = self.aus.min(axis=0)
		maxs = self.aus.max(axis=0)
		X,Y = np.meshgrid(linspace(mins[0], maxs[0], self.quiver_res), 
								linspace(mins[1], maxs[1], self.quiver_res))
		Z = np.dstack([X,Y])
		vals = self.f(0,Z.transpose(2,0,1))
		PL.quiver(X,Y,vals[0], vals[1])


class ExplicitEuler (ODESolver):
	def step(self, t, u):
		return t + self.h, u + self.h*self.f(t, u)

class ImplicitEuler (ODESolver):
	def step(self, t, u):
		res = self.f.res(u, t, self.h)
		return t + self.h, res


class RungeKutta4 (ODESolver):
	"""
	Runge-Kutta of order 4.
	"""
	def step(self, t, u):
		f = self.f
		h = self.h
		Y1 = f(t, u)
		Y2 = f(t + h/2., u + h*Y1/2.)
		Y3 = f(t + h/2., u + h*Y2/2.)
		Y4 = f(t + h, u + h*Y3)
		return t+h, u + h/6.*(Y1 + 2.*Y2 + 2.*Y3 + Y4)

class RungeKutta34 (ODESolver):
	"""
	Adaptive Runge-Kutta of order four.
	"""
	error_order = 4.
	# default tolerance
	tol = 1e-6

	def increment_stepsize(self):
		self.h *= (self.tol/self.error)**(1/self.error_order)

	def step(self, t, u):
		f = self.f
		h = self.h
		Y1 = f(t, u)
		Y2 = f(t + h/2., u + h*Y1/2.)
		Y3 = f(t + h/2, u + h*Y2/2)
		Z3 = f(t + h, u - h*Y1 + 2*h*Y2)
		Y4 = f(t + h, u + h*Y3)
		self.error = norm(h/6*(2*Y2 + Z3 - 2*Y3 - Y4))
		return t+h, u + h/6*(Y1 + 2*Y2 + 2*Y3 + Y4)

class ode15s(ODESolver):
	"""
	Simulation of matlab's ode15s solver.
	It is a BDF method of max order 5
	"""
	def __init__(self, *args, **kwargs):
		super(ode15s, self).__init__(*args, **kwargs)
		self.setup()

	def setup(self, **kw):
		import scipy.integrate
		self.integ = scipy.integrate.ode(self.f)
		self.integ.set_integrator('vode', method='bdf', order=5, nsteps=3000, **kw)
	
	def initialize(self, *args, **kwargs):
		super(ode15s, self).initialize(*args, **kwargs)
		self.integ.set_initial_value(self.us[0], self.ts[0])

	def step(self, t, u):
		self.integ.integrate(self.integ.t + self.h)
		if not self.integ.successful():
			print("vode error")
		return self.integ.t, self.integ.y

class McLachlan(ODESolver):
	def __init__(self, system):
		self.system = system
	
	def step(self, t, u):
		h = self.h
		vel = self.system.velocity(u)
		qh = self.system.state(u) + .5*self.h*vel
		force = self.system.force(qh)
		codistribution = self.system.codistribution(qh)
		lag = self.system.lag(u)
		def residual(vl):
			v,l = self.system.vel_lag_split(vl)
			return np.hstack([v - vel + h * (force + dot(codistribution.T, l)), dot(codistribution, v)])
		N = Newton(residual)
		vl = N.run(self.system.vel_lag_stack(vel, lag))
		vnew, lnew = self.system.vel_lag_split(vl)
		qnew = qh + .5*h*vnew
		return t+h, self.system.assemble(qnew,vnew,lnew)
			

class ContactOscillator(object):
	def __init__(self, epsilon=0):
		self.epsilon = epsilon
	
	def state(self, u):
		return u[:3]
	
	def velocity(self, u):
		return u[3:6]
	
	def lag(self, u):
		return u[6:7]
		
	def assemble(self, q,v,l):
		return np.hstack([q,v,l])
	
	def vel_lag_split(self, vl):
		return vl[:3], vl[3:4]
	
	def vel_lag_stack(self, v,l):
		return np.hstack([v,l])
	
	def force(self, q):
		return -q - self.epsilon*q[2]*q[0]*array([q[2],0,q[0]])
	
	def codistribution(self, q):
		return array([[1., 0, q[1]]])

class Test_McOsc(object):
	def setUp(self):
		self.sys = ContactOscillator()
		self.s = McLachlan(self.sys)
		self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
	
	def test_run(self):
		self.s.run()
		
if __name__ == '__main__':
	t = Test_McOsc()
	t.setUp()
	t.test_run()