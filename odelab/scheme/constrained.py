# -*- coding: UTF-8 -*-
from __future__ import division

import numpy as np

from odelab.scheme import Scheme

from odelab.scheme.rungekutta import *

import odelab.newton as _rt

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

	
	root_solver = _rt.Newton
	
	def step(self, t, u):
		h = self.h
		vel = self.system.velocity(u)
		qh = self.system.position(u) + .5*self.h*vel
		force = self.system.force(qh)
		codistribution = self.system.codistribution(qh)
		lag = self.system.lag(u)
		def residual(vl):
			v,l = self.system.vel_lag_split(vl)
			return np.hstack([v - vel - h * (force + np.dot(codistribution.T, l)), np.dot(self.system.codistribution(qh+.5*h*v), v)])
		N = self.root_solver(residual)
		vl = N.run(self.system.vel_lag_stack(vel, lag))
		vnew, lnew = self.system.vel_lag_split(vl)
		qnew = qh + .5*h*vnew
		return t+h, self.system.assemble(qnew,vnew,lnew)
	
	



class Spark(Scheme):
	r"""
Solver for semi-explicit index 2 DAEs by Lobatto III RK methods 
 as presented in [jay03]_.
 
We consider the following system of DAEs:

.. math::
    y' = f(t,y,z)\\
     0 = g(t,y)
 
where t is the independent variable, y is a vector of length n containing
the differential variables and z is a vector of length m containing the 
algebraic variables. Also,

.. math::
 
    f:\ R × R^n × R^m → R^n\\
    g:\ R × R^n → R^m
 
It is assumed that :math:`g_y f_z` exists and is invertible.

The above system is integrated from ``t0`` to ``tfinal``, where 
tspan = [t0, tfinal] using constant stepsize h. The initial condition is 
given by ``(y,z) = (y0,z0)`` and the number of stages in the Lobatto III RK 
methods used is given by ``s``.
 
 
 
The set of nonlinear SPARK equations are solved using the solver in :attr:`root_solver`.
 
 
 References:
	
.. [jay03] \L. Jay - Solution of index 2 implicit differential-algebraic equations
	    by Lobatto Runge-Kutta methods (2003).
	"""
	
	root_solver = _rt.FSolve

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
		Q = np.dot(L,Q)
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
			dyn = [np.dot(vec, RK_class.tableaux[s][:,1:].T) for RK_class, vec in dyn_dict.items()]
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
