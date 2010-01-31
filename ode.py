# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np
from numpy import array, dot
from numpy.linalg import norm, inv
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
		return jacobian(self.residual, x, self.h)
	
	def residual(self, x):
		a = x.reshape(self.shape)
		res = self.F(a)
		try:
			res_vec = res.ravel()
		except AttributeError: # a list of arrays
			res_vec = np.hstack([comp.ravel() for comp in res])
		return res_vec
	
	maxiter = 300
	tol = 1e-8
	def run(self, x0):
		"""
		Run the Newton iteration.
		"""
		if np.isscalar(x0):
			x = array([x0])
		else:
			x = array(x0)
		self.shape = x.shape
		x = x.ravel()
		for i in xrange(self.maxiter):
			d = self.der(x)
			y = self.level - self.residual(x)
			if np.isscalar(y):
				incr = y/d.item()
			else:
				try:
					incr = np.linalg.solve(d, y)
				except np.linalg.LinAlgError, ex: # should use the "as" syntax!!
					eigvals, eigs = np.linalg.eig(d)
					zerovecs = eigs[:, np.abs(eigvals) < 1e-10]
					raise np.linalg.LinAlgError("%s: %s" % (ex.message, repr(zerovecs)))
			x += incr
			if norm(incr) < self.tol:
				break
##			if self.is_zero(x):
##				break
		else:
			raise Exception("Newton algorithm did not converge. ∆x=%.2e" % norm(incr))
		self.required_iter = i
		return x.reshape(self.shape)
	
	def is_zero(self, x):
		res = norm(self.F(x) - self.level)
		return res < self.tol

import numpy.testing as npt
import nose.tools as nt

class Test_Newton(object):
	
	dim = 10
	
	def test_run(self):
		zr = np.random.rand(self.dim)
		def f(x):
			return np.arctan(x - zr)
		xr = np.zeros(self.dim)
		n = Newton(f)
		x = n.run(xr)
		print n.required_iter
		npt.assert_array_almost_equal(x,zr)
	
	def test_scalar(self):
		def f(x):
			return (x-1)**2
		n = Newton(f)
		n.tol = 1e-9
		z = n.run(10.)
		npt.assert_almost_equal(z,1.)
		
	def test_copy(self):
		"""Newton doesn't destroy the initial value"""
		def f(x):
			return x
		N = Newton(f)
		x0 = array([1.])
		y0 = x0.copy()
		N.run(x0)
		nt.assert_true(x0 is not y0)
		npt.assert_almost_equal(x0,y0)
	
	def test_array(self):
		def f(a):
			return a*a
		expected = np.zeros([2,2])
		N = Newton(f)
		x0 = np.ones([2,2])
		res = N.run(x0)
		npt.assert_almost_equal(res, expected)

class ODESolver (object):

	def __init__(self, system=None):
		self.system = system
	
	@property
	def f(self):
		return self.system.f

	def increment_stepsize(self):
		"""
		Change the step size based on error estimation.
		To be overridden for a variable step size method.
		"""
		pass

	max_iter = 100000
	
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
		self.aus = array(self.us).T


	def plot(self, components=None):
		"""
		Plot some components of the solution.
		"""
		if components is None:
			components = range(len(self.us[:,0]))
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
	
	def step(self, t, u):
		h = self.h
		vel = self.system.velocity(u)
		qh = self.system.position(u) + .5*self.h*vel
		force = self.system.force(qh)
		codistribution = self.system.codistribution(qh)
		lag = self.system.lag(u)
		def residual(vl):
			v,l = self.system.vel_lag_split(vl)
			return np.hstack([v - vel - h * (force + dot(codistribution.T, l)), dot(self.system.codistribution(qh+.5*h*v), v)])
		N = Newton(residual)
		vl = N.run(self.system.vel_lag_stack(vel, lag))
		vnew, lnew = self.system.vel_lag_split(vl)
		qnew = qh + .5*h*vnew
		return t+h, self.system.assemble(qnew,vnew,lnew)
	
	def plot_qv(self, i=2, skip=1, *args, **kwargs):
		qs = self.system.position(self.aus)
		vs = self.system.velocity(self.aus)
		plot(qs[i,::skip], vs[i,::skip], *args, **kwargs)
	
	def plot_H(self, *args, **kwargs):
		plot(self.ats, self.system.energy(self.aus), *args, **kwargs)

class RungeKutta(ODESolver):
	pass

class LobattoIIIA(RungeKutta):
	
	sf = np.sqrt(5)
	tableaux = {
	2: array([	[0.,0.],
				[.5,.5],
				[.5,.5]]),
	3: array([	[0.,0.,0.],
				[5/24,1/3,-1/24],
				[1/6,2/3,1/6],
				[1/6,2/3,1/6]]),
	4: array([	[0., 0.,0.,0.],
				[(11+sf)/120, (25-sf)/120,    (25-13*sf)/120, (-1+sf)/120],
				[(11-sf)/120, (25+13*sf)/120, (25+sf)/120, (-1-sf)/120],
				[1/12,             5/12,                5/12,                1/12],
				[1/12, 5/12, 5/12, 1/12]])
	}

class LobattoIIIB(RungeKutta):
	sf = np.sqrt(5)
	tableaux = {	
	2: array([	[1/2, 0],
				[1/2, 0],
				[1/2, 1/2]]),
				
	3: array([	[1/6, -1/6, 0],
				[1/6,  1/3, 0],
				[1/6,  5/6, 0],
				[1/6, 2/3, 1/6]]),
				
	4: array([	[1/12, (-1-sf)/24,     (-1+sf)/24,     0],
				[1/12, (25+sf)/120,    (25-13*sf)/120, 0],
				[1/12, (25+13*sf)/120, (25-sf)/120,    0],
				[1/12, (11-sf)/24,    (11+sf)/24,    0],
				[1/12, 5/12, 5/12, 1/12]])
	}

class LobattoIIIC(RungeKutta):
	sf = np.sqrt(5)
	tableaux = {
2: array([
[1/2, -1/2],
[1/2,  1/2],
[1/2, 1/2]]),
3: array([
[1/6, -1/3,   1/6],
[1/6,  5/12, -1/12],
[1/6,  2/3,   1/6],
[1/6, 2/3, 1/6]]),
4: array([
[1/12, -sf/12,       sf/12,        -1/12],
[1/12, 1/4,               (10-7*sf)/60, sf/60],
[1/12, (10+7*sf)/60, 1/4,               -sf/60],
[1/12, 5/12,              5/12,              1/12],
[1/12, 5/12, 5/12, 1/12]])
}

class LobattoIIICs(RungeKutta):
	tableaux = {
2: array([
[0., 0],
[1, 0],
[1/2, 1/2]]),
3: array([
[0,   0,   0],
[1/4, 1/4, 0],
[0,   1,   0],
[1/6, 2/3, 1/6]])
	}

class LobattoIIID(RungeKutta):
	tableaux = {
2: array([
[1/4, -1/4],
[3/4, 1/4],
[1/2, 1/2]]),
3: array([
[1/12, -1/6,  1/12],
[5/24,  1/3, -1/24],
[1/12,  5/6,  1/12],
[1/6, 2/3, 1/6]])
	}

class Spark(ODESolver):
	
## 	RK_classes = [LobattoIIIA, LobattoIIIB, LobattoIIIC, LobattoIIICs, LobattoIIID]
	
	def __init__(self, system, nb_stages):
		self.system = system
		self.nb_stages = nb_stages
	
	def Q(self):
		s = self.nb_stages
		A1t = LobattoIIIA.tableaux[s][1:-1]
		es = np.zeros(s)
		es[-1] = 1.
		Q = np.zeros([s,s+1])
		Q[:-1,:-1] = A1t
		Q[-1,-1] = 1.
		L = np.linalg.inv(np.vstack([A1t, es]))
		Q = dot(L,Q)
		return Q
	
## 	def simpleQ(self):
## 		s = self.nb_stages
## 		Q = np.zeros([s,s+1])
## 		Q[:s,:s] = np.identity(s)
## 		return Q
## 		
	
	def step(self, t, u):
		s = self.nb_stages
		h = self.h
## 		As = np.dstack([RK.tableaux[s] for RK in self.RK_classes])
## 		T = As[:-1].sum(1)
## 		ts = t + T
		Q = self.Q()
		y = self.system.state(u).copy()
		yc = y.reshape(-1,1) # "column" vector
		
		def residual(YZ):
## 			f = np.dstack(self.system.dynamics(t, YZ[:,:-1])) # should not be t!
			dyn_dict = self.system.dynamics(t, YZ[:,:-1])
			Y = self.system.state(YZ)
## 			r1 = Y - yc - h*np.tensordot(As, f, axes=[[2,1], [2,1]]).T
			r1 = Y - yc - h*sum(dot(vec, RK_class.tableaux[s].T) for RK_class, vec in dyn_dict.items())
			r2 = np.dot(self.system.constraint(t, YZ), Q.T)
			# unused final versions of z
			r3 = self.system.lag(YZ[:,-1])
			return [r1,r2,r3]
		
		N = Newton(residual)
## 		guess = np.random.rand(len(u),s+1)
		guess = np.column_stack([u.copy()]*(s+1))
		result = N.run(guess)
		return t+h, result[:,-1]
			

class JayExample(object):
	def dynamics(self, t, u):
		y1,y2,z = u
		return {
			LobattoIIIA: array([y2 - 2*y1**2*y2, -y1**2]),
			LobattoIIIB: array([y1*y2**2*z**2, np.exp(-t)*z - y1]),
			LobattoIIIC: array([-y2**2*z, -3*y2**2*z]),
			LobattoIIICs: array([2*y1*y2**2 - 2*np.exp(2*t)*y1*y2]),
			LobattoIIID: array([2*y2**2*z**2, y1**2*y2**2])
			}
	
	def constraint(self, t, u):
		return array([u[0]**2*u[1] -  1])
	
	def state(self, u):
		return u[:2]
	
	def lag(self, u):
		return u[2:3]

class GraphSystem(object):
	"""
	Trivial semi-explicit index 2 DAE of the form::
		x' = 1
		y' = lambda
		y = f(x)
	"""
	def __init__(self, f):
		self.f = f
	
	def state(self, u):
		return u[:2]
	
	def lag(self, u):
		return u[-1:]
	
	def dynamics(self, t, u):
		x,y = self.state(u)
		return {
			LobattoIIIA: np.zeros_like(self.state(u)), 
			LobattoIIIB: array([np.ones_like(x), self.lag(u)[0]]),
			}
	
	def constraint(self, t, u):
		x,y = self.state(u)
		return array([y - self.f(x)])
	
	def hidden_error(self, t, u):
		return self.lag(u)[0] - self.f.der(t,self.state(u)[0])

class ODESystem(object):
	def __init__(self, f, RK_class=LobattoIIIA):
		self.f = f
		self.RK_class = RK_class
	
	def state(self, u):
		return u[:1]
	
	def lag(self,u):
		return u[1:] # should be empty
	
	def dynamics(self, t, u):
		return {self.RK_class: array([self.f(u)])}
	
	def constraint(self, t, u):
		return np.zeros([0,np.shape(u)[1]])
	
	

class ContactOscillator(object):
	def __init__(self, epsilon=0.):
		self.epsilon = epsilon
	
	def position(self, u):
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


class Test_McOsc(object):
	def setUp(self):
		self.sys = ContactOscillator()
		self.s = McLachlan(self.sys)
		self.s.initialize(array([1.,1.,1.,0.,0,0,0]))
		self.s.time = 10.
	
	def test_run(self):
		self.s.run()
	
	def test_H(self):
		self.s.run()
		self.s.plot_H()
	
	z0s = np.linspace(-.9,.9,10)
	N = 40
	
	def test_z0(self, i=5):
		z0 = self.z0s[i]
		self.s.initialize(u0=self.sys.initial(z0), h=self.sys.time_step(self.N))
		self.s.time = self.N*self.s.h
		self.s.run()
		
class Test_SparkODE(object):
	def setUp(self):
		def f(x):
			return -x
		self.sys = ODESystem(f)
		self.s = Spark(self.sys, 2)
		self.s.initialize(array([1.]))
	
	def test_run(self):
		self.s.run()
		exact = np.exp(-self.s.ats).reshape(1,-1)
		print exact[:,-1]
		print self.s.us[-1]
		npt.assert_array_almost_equal(self.s.aus, exact, 5)
## 		plot(self.s.ats, np.vstack([self.s.aus, exact]).T)

class Test_Jay(object):
	def setUp(self):
		def sq(x):
			return x*x
## 		self.sys = GraphSystem(sq)
		self.sys = JayExample()
		self.s = Spark(self.sys, 2)
		self.s.initialize(array([1.,1.,0]))
## 		self.s.initialize(array([1.]))
	
	def test_run(self):
		self.s.run()

if __name__ == '__main__':
	t = Test_Jay()
	t.setUp()
	t.test_run()
## 	t.test_z0(1)
## 	t.s.plot_qv(1)