# -*- coding: UTF-8 -*-
"""
:mod:`Solver` -- ODE Solvers
============================

A collection of solvers for ODEs of various types.

.. module :: solver
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

The :class:`Scheme` class contains methods on how to perform one iteration step. 
It is its responsibility to take care of the time step.

The higher level class is :class:`Solver`, which is initialized with an instance of a :class:`Scheme` class.
"""
from __future__ import division

import numpy as np
from numpy import array, dot
from numpy.linalg import norm, inv
import pylab as PL
from pylab import plot, legend

from odelab.newton import Newton, FSolve

import itertools

class SimulationInfo(object):
	pass

class Simulator(object):
	def __init__(self, solver, time=None):
		self.solver = solver
	


class Solver (object):
	"""
	General Solver class, that takes care of calling the step function and storing the intermediate results.
	
	:Parameters:
		system : :class:`System`
			Object describing the system. The requirement on that class may vary. See the documentation of the various solver subclasses. The system may also be specified later, although before any simulation of course.
	"""

	def __init__(self, system=None):
		self.system = system
	
	
	# default values for the total time
	time = 1.

	def initialize(self, u0=None, t0=0, h=None, time=None):
		"""
		Initialize the solver to the initial condition :math:`u(t0) = u0`.
		
		:Parameters:
			u0 : array
				initial condition; if it is not provided, it is set to the previous initial condition.
			t0 : scalar
				initial time
			h : scalar
				time step
			time : scalar
				time span of the simulation
		"""
		if u0 is not None:
			if np.isscalar(u0):
				u0 = [u0]
			u0 = np.array(u0)
			u0 = self.system.preprocess(u0)
		else: # start from the previous initial conditions
			u0 = self.us[0]
			t0 = self.ts[0]
		self.ts = [t0]
		self.us = [u0]
		if h is not None:
			self.h = h
		if time is not None:
			self.time = time
		self.initialize_scheme()

	def initialize_scheme(self):
		pass

	def generate(self, t, u):
		"""
		Generates the (t,u) values.
		"""
		for i in itertools.count(): # infinite loop
			t, u = self.scheme.step(t, u)
			yield t, u
			self.scheme.increment_stepsize()
	

	max_iter = 100000
	class FinalTimeNotReached(Exception):
		pass
	
	def simulating(self):
		return self
	
	def __enter__(self):
		sim_info = SimulationInfo()
		# start from the last time we stopped
		t = t0 = self.ts[-1]
		u = self.us[-1]
		sim_info.qs = []
		sim_info.generator = self.generate(t, u)
		self.sim_info = sim_info
		return sim_info
		
	def __exit__(self, ex_type, ex_value, traceback):
		self.ts.extend(q[0] for q in self.sim_info.qs)
		self.us.extend(q[1] for q in self.sim_info.qs)

		self.ats = array(self.ts)
		self.aus = array(self.us).T

	t_tol = 1e-12 # tolerance to tell whether the final time is reached
	
	def run(self, time=None):
		"""
		Run the simulation for a given time.
		
		:Parameters:
			time : scalar
				the time span for which to run; if none is given, the default ``self.time`` is used
		"""
		if time is None:
			time = self.time
		t0 = self.ts[0]
		tf = t0 + time # final time
		with self as sim_info:
			for i in xrange(self.max_iter):
				t,u = next(sim_info.generator)
				self.sim_info.qs.append((t,u))
				if t > tf - self.t_tol:
					break
			else:
				raise self.FinalTimeNotReached("Reached maximal number of iterations: {0}".format(self.max_iter))

	
	
	def plot(self, components=None):
		"""
		Plot some components of the solution.
		
		:Parameters:
			components : scalar|array_like
				either a given component of the solution, or a list of components to plot.
		"""
		if components is None:
			components = range(len(self.us[0]))
		if not np.iterable(components):
			components = [components]
		has_exact = hasattr(self.system, 'exact')
		if has_exact:
			exact = self.system.exact(self.ats)
		for component in components:
			label = self.system.label(component)
			plot(self.ats, self.aus[component], ',-', label=label)
			if has_exact:
				PL.gca()._get_lines.count -= 1
				plot(self.ats, exact[component], ls='-', lw=2, label='%s_' % label)
		PL.xlabel('time')
		PL.legend()
	
	def plot_function(self, function):
		"""
		Plot a given function of the state. May be useful to plot constraints or energy.
		
		:Parameters:
			function : string
				name of the method to call on the current system object.
		
		:Example:
			the code::
			
				solver.plot_function('energy')
			
			will call the method ``solver.system.energy`` on the current stored solution points.
		"""
		values = self.system.__getattribute__(function)(np.vstack([self.aus, self.ats]))
		plot(self.ats, values.T)

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

class SingleStepSolver(Solver):
	def __init__(self, scheme, system):
		super(SingleStepSolver, self).__init__(system)
		self.set_scheme(scheme)

	def initialize_scheme(self):
		self.scheme.initialize()
	
	def set_scheme(self, scheme):
		self.scheme = scheme
		scheme.solver = self

class Scheme(object):

	@property
	def system(self):
		return self.solver.system

	def increment_stepsize(self):
		"""
		Change the step size based on error estimation.
		To be overridden for a variable step size method.
		"""
		pass
	
	h_default = .01
	
	# to be removed; use a signal instead
	def get_h(self):
		return self._h
	def set_h(self, h):
		self._h = h
		self._h_dirty = True
	h = property(get_h, set_h)

	def initialize(self):
		try:
			self.h = self.solver.h
		except AttributeError:
			self.h = self.h_default

class ExplicitEuler (Scheme):
	def step(self, t, u):
		return t + self.h, u + self.h*self.system.f(t, u)

class ImplicitEuler (Scheme):
	def step(self, t, u):
		res = self.system.f.res(u, t, self.h)
		return t + self.h, res


class RungeKutta4 (Scheme):
	"""
	Runge-Kutta of order 4.
	"""
	def step(self, t, u):
		f = self.system.f
		h = self.h
		Y1 = f(t, u)
		Y2 = f(t + h/2., u + h*Y1/2.)
		Y3 = f(t + h/2., u + h*Y2/2.)
		Y4 = f(t + h, u + h*Y3)
		return t+h, u + h/6.*(Y1 + 2.*Y2 + 2.*Y3 + Y4)

class RungeKutta34 (Scheme):
	"""
	Adaptive Runge-Kutta of order four.
	"""
	error_order = 4.
	# default tolerance
	tol = 1e-6

	def increment_stepsize(self):
		if self.error > 1e-15:
			self.h *= (self.tol/self.error)**(1/self.error_order)
		else:
			self.h = 1.

	def step(self, t, u):
		f = self.system.f
		h = self.h
		Y1 = f(t, u)
		Y2 = f(t + h/2., u + h*Y1/2.)
		Y3 = f(t + h/2, u + h*Y2/2)
		Z3 = f(t + h, u - h*Y1 + 2*h*Y2)
		Y4 = f(t + h, u + h*Y3)
		self.error = norm(h/6*(2*Y2 + Z3 - 2*Y3 - Y4))
		return t+h, u + h/6*(Y1 + 2*Y2 + 2*Y3 + Y4)

class ode15s(Scheme):
	"""
	Simulation of matlab's ``ode15s`` solver.
	It is a BDF method of max order 5
	"""
	
	
	def __init__(self, **kwargs):
		self.integrator_kwargs = kwargs

	def initialize(self): # the system must be defined before this is called!
		super(ode15s,self).initialize()
		import scipy.integrate
		self.integ = scipy.integrate.ode(self.system.f)
		vodevariant = ['vode', 'zvode'][np.iscomplexobj(self.solver.us[0])]
		self.integ.set_integrator(vodevariant, method='bdf', order=5, nsteps=3000, **self.integrator_kwargs)
		self.integ.set_initial_value(self.solver.us[0], self.solver.ts[0])

	def step(self, t, u):
		self.integ.integrate(self.integ.t + self.h)
		if not self.integ.successful():
			print("vode error")
		return self.integ.t, self.integ.y

# exponential integrators

from phi_pade import Phi, Polynomial

class Exponential(Scheme):
	"""
	Explicit Exponential Integrator Class.
	"""
	def __init__(self, *args, **kwargs):
		super(Exponential, self).__init__(*args, **kwargs)
		self.phi = Phi(self.phi_order, self.phi_degree)
	
	phi_degree = 6
	
	def initialize(self):
		super(Exponential, self).initialize()
		self.tail = self.solver.us[-1].reshape(-1,1)
	
	def step(self, t, u):
		h = self.h
		ua, vb = self.general_linear()
		nb_stages = len(ua)
		nb_steps = len(vb)
		Y = np.zeros([len(u), nb_stages+nb_steps], dtype=u.dtype)
		Y[:,-nb_steps:] = self.tail
		newtail = np.zeros_like(self.tail) # alternative: work directly on self.tail
		for s in range(nb_stages):
			uas = ua[s]
			for j, coeff in enumerate(uas[1:]):
				if coeff is not None:
					Y[:,s] += np.dot(coeff, Y[:,j])
			Y[:,s] = h*self.system.nonlin(t+uas[0]*h, Y[:,s])
			if np.any(np.isnan(Y[:,s])):
				raise Exception('unstable %d' % s)
		for r in range(nb_steps):
			vbr = vb[r]
			for j, coeff in enumerate(vbr):
				if coeff is not None:
					newtail[:,r] += np.dot(coeff, Y[:,j])
		self.tail = newtail
		return t + h, self.tail[:,0]
		

	def general_linear(self):
		"""
		Currently returns a cached version of the coefficients of the method.
		"""
		if self._h_dirty: # recompute the coefficients if h had changed
			z = self.h * self.system.linear()
			self._general_linear = self.general_linear_z(z)
			self._h_dirty = False
		return self._general_linear

class LawsonEuler(Exponential):
	phi_order = 0
	
	def general_linear_z(self, z):
		ez = self.phi(z)[0]
		one = Polynomial.exponents(z,0)[0]
		return [[0., None, one]], [[ez, ez]]

class RKMK4T(Exponential):
	phi_order = 1
	
	def general_linear_z(self, z):
		one = Polynomial.exponents(z,0)[0]
		ez, phi_1 = self.phi(z)
		ez2, phi_12 = self.phi(z/2)
		return ([	[0, None, None, None, None, one],
					[1/2, 1/2*phi_12, None, None, None, ez2],
					[1/2, 1/8*np.dot(z, phi_12), 1/2*np.dot(phi_12,one-1/4*z), None, None, ez2],
					[1, None, None, phi_1, None, ez]
				],
				[[1/6*np.dot(phi_1,one+1/2*z), 1/3*phi_1, 1/3*phi_1, 1/6*np.dot(phi_1,one-1/2*z), ez]]
				)
		



class McLachlan(Scheme):
	"""
Solver for the Lagrange-d'Alembert (LDA) equations using the
algorithm given by equation (4.18) in [mclachlan06]_.
 
 The Lagrangian is assumed to be of the form:
 
.. math::

    L(q,v) = 0.5 \|v^2\| - V(q)
 
 where :math:`V(q)` is the potential energy. The constraints are given by :math:`Av=0`, 
 where :math:`A` is the mxn constraint matrix.
 
 
:References:
	
.. [mclachlan06] \R. McLachlan and M. Perlmutter, *Integrators for Nonholonomic Mechanical Systems*, J. Nonlinear Sci., **16** 283-328, (2006) (`url <http://dx.doi.org/10.1007/s00332-005-0698-1>`_)
	"""

	
	root_solver = Newton
	
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
		N = self.root_solver(residual)
		vl = N.run(self.system.vel_lag_stack(vel, lag))
		vnew, lnew = self.system.vel_lag_split(vl)
		qnew = qh + .5*h*vnew
		return t+h, self.system.assemble(qnew,vnew,lnew)
	
	



class Spark(Scheme):
	"""
Solver for semi-explicit index 2 DAEs by Lobatto III RK methods 
 as presented in [jay03]_.
 
 We consider the following system of DAEs
 
    y' = f(t,y,z)
     0 = g(t,y)
 
 where t is the independent variable, y is a vector of length n containing
 the differential variables and z is a vector of length m containing the 
 algebraic variables. Also,
 
    f: R x R^n x R^m -> R^n
    g: R x R^n -> R^m
 
 It is assumed that $g_y f_z$ exists and is invertible.
 
 The above system is integrated from t0 to tfinal, where 
 tspan = [t0, tfinal] using constant stepsize h. The initial condition is 
 given by (y,z) = (y0,z0) and the number of stages in the Lobatto III RK 
 methods used is given by s.
 
 
 
 The set of nonlinear SPARK equations are solved using the solver in ``root_solver``.
 
 
 References:
	
.. [jay03] \L. Jay - Solution of index 2 implicit differential-algebraic equations
	    by Lobatto Runge-Kutta methods (2003).
	"""
	
	root_solver = FSolve

	def __init__(self, nb_stages):
		super(Spark, self).__init__()
		self.nb_stages = nb_stages
	
	def Q(self):
		s = self.nb_stages
		A1t = LobattoIIIA.tableaux[s][1:-1,1:]
		es = np.zeros(s)
		es[-1] = 1.
		Q = np.zeros([s,s+1])
		Q[:-1,:-1] = A1t
		Q[-1,-1] = 1.
		L = np.linalg.inv(np.vstack([A1t, es]))
		Q = dot(L,Q)
		return Q
	
	def get_residual_function(self, t, u):
		s = self.nb_stages
		h = self.h
		c = LobattoIIIA.tableaux[s][:,0]
		T = t + c*h
		Q = self.Q()
		y = self.system.state(u).copy()
		yc = y.reshape(-1,1) # "column" vector
		
		def residual(YZ):
			YZT = np.vstack([YZ,T])
			dyn_dict = self.system.multi_dynamics(YZT[:,:-1])
			Y = self.system.state(YZ)
			dyn = [dot(vec, RK_class.tableaux[s][:,1:].T) for RK_class, vec in dyn_dict.items()]
			r1 = Y - yc - h*sum(dyn)
			r2 = np.dot(self.system.constraint(YZT), Q.T)
			# unused final versions of z
			r3 = self.system.lag(YZ[:,-1])
			return [r1,r2,r3]
		
		return residual

	def step(self, t, u):
		s = self.nb_stages
		residual = self.get_residual_function(t, u)
		guess = np.column_stack([u.copy()]*(s+1))
		N = self.root_solver(residual)
		residual(guess)
		full_result = N.run(guess) # all the values of Y,Z at all stages
		# we keep the last Z value:
		result = np.hstack([self.system.state(full_result[:,-1]), self.system.lag(full_result[:,-2])])
		return t+self.h, result
	

# Coefficients for the Lobatto methods		

class RungeKutta(Scheme):
	"""
	Collection of classes containing the coefficients of various Runge-Kutta methods.
	
	:Attributes:
		tableaux : dictionary
			dictionary containing a Butcher tableau for every available number of stages.
	"""

class LobattoIIIA(RungeKutta):
	
	sf = np.sqrt(5)
	tableaux = {
	2: array([	[0., 0.,0.],
				[1., .5,.5],
				[1, .5,.5]]),
	3: array([	[0  ,0.,0.,0.],
				[1/2,5/24,1/3,-1/24],
				[1  ,1/6,2/3,1/6],
				[1 ,1/6,2/3,1/6]]),
	4: array([	[0        ,0., 0.,0.,0.],
				[(5-sf)/10,(11+sf)/120, (25-sf)/120,    (25-13*sf)/120, (-1+sf)/120],
				[(5+sf)/10,(11-sf)/120, (25+13*sf)/120, (25+sf)/120, (-1-sf)/120],
				[1        ,1/12,             5/12,                5/12,                1/12],
				[1       ,1/12, 5/12, 5/12, 1/12]])
	}

class LobattoIIIB(RungeKutta):
	sf = np.sqrt(5)
	tableaux = {	
	2: array([	[0.,1/2, 0],
				[1.,1/2, 0],
				[1,1/2, 1/2]]),
				
	3: array([	[0  ,1/6, -1/6, 0],
				[1/2,1/6,  1/3, 0],
				[1  ,1/6,  5/6, 0],
				[1 ,1/6, 2/3, 1/6]]),
				
	4: array([	[0        ,1/12, (-1-sf)/24,     (-1+sf)/24,     0],
				[(5-sf)/10,1/12, (25+sf)/120,    (25-13*sf)/120, 0],
				[(5+sf)/10,1/12, (25+13*sf)/120, (25-sf)/120,    0],
				[1        ,1/12, (11-sf)/24,    (11+sf)/24,    0],
				[1       ,1/12, 5/12, 5/12, 1/12]])
	}

class LobattoIIIC(RungeKutta):
	sf = np.sqrt(5)
	tableaux = {
2: array([
[0.,1/2, -1/2],
[1.,1/2,  1/2],
[1,1/2, 1/2]]),
3: array([
[0  ,1/6, -1/3,   1/6],
[1/2,1/6,  5/12, -1/12],
[1  ,1/6,  2/3,   1/6],
[1 ,1/6, 2/3, 1/6]]),
4: array([
[0        ,1/12, -sf/12,       sf/12,        -1/12],
[(5-sf)/10,1/12, 1/4,               (10-7*sf)/60, sf/60],
[(5+sf)/10,1/12, (10+7*sf)/60, 1/4,               -sf/60],
[1        ,1/12, 5/12,              5/12,              1/12],
[1       ,1/12, 5/12, 5/12, 1/12]])
}

class LobattoIIICs(RungeKutta):
	tableaux = {
2: array([
[0.,0., 0],
[1.,1, 0],
[1,1/2, 1/2]]),
3: array([
[0  ,0,   0,   0],
[1/2,1/4, 1/4, 0],
[1  ,0,   1,   0],
[1 ,1/6, 2/3, 1/6]])
	}

class LobattoIIID(RungeKutta):
	tableaux = {
2: array([
[0.,1/4, -1/4],
[1.,3/4, 1/4],
[1,1/2, 1/2]]),
3: array([
[0  ,1/12, -1/6,  1/12],
[1/2,5/24,  1/3, -1/24],
[1  ,1/12,  5/6,  1/12],
[1 ,1/6, 2/3, 1/6]])
	}

