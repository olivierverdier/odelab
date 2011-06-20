## -*- coding: UTF-8 -* -*- coding: UTF-8 -*-

from __future__ import division

import numpy as np

from odelab.scheme import Scheme

from odelab.scheme.rungekutta import *

import odelab.newton as _rt

class McLachlan(Scheme):
	"""
Solver for the Lagrange-d'Alembert (LDA) equations using the
algorithm given by equation (4.18) in [mclachlan06]_.

 The Lagrangian is assumed to be *separable*:

.. math::

    L(q,v) = T(v) - V(q)

where :math:`T(v)` is the kinetic energy and :math:`V(q)` is the potential energy.
The constraints are given by :math:`Av=0`,
where :math:`A` is the mxn constraint matrix.

More precisely, the :class:`odelab.system.System` object must implement:

* :meth:`odelab.system.System.velocity`
* :meth:`odelab.system.System.momentum`
* :meth:`odelab.system.System.position`
* :meth:`odelab.system.System.lag`
* :meth:`odelab.system.System.force`
* :meth:`odelab.system.System.codistribution`


:References:

.. [mclachlan06] \R. McLachlan and M. Perlmutter, *Integrators for Nonholonomic Mechanical Systems*, J. Nonlinear Sci., **16** 283-328, (2006) :doi:`/10.1007/s00332-005-0698-1>`
	"""

	root_solver = _rt.Newton

	def step(self, t, u, h):
		v0 = self.system.velocity(u)
		momentum = self.system.momentum
		p0 = momentum(u)
		qh = self.system.position(u) + .5*h*v0
		force = self.system.force(qh)
		codistribution = self.system.codistribution
		codistribution_h = codistribution(qh)
		def residual(u1):
			q1 = self.system.position(u1)
			v1 = self.system.velocity(u1)
			l = self.system.lag(u1)
			return np.hstack([q1 - qh - .5*h*v1, momentum(u1) - p0 - h * (force + np.dot(codistribution_h.T, l)), np.dot(codistribution(q1), v1)])
		N = self.root_solver(residual)
		unew = N.run(u)
		return t+h, unew

class NonHolonomicEnergy(Scheme):

	root_solver = _rt.Newton

	def step(self, t, u, h):
		v0 = self.system.velocity(u)
		q0 = self.system.position(u)
		#qh = q + .5*self.h*vel
		#force = self.system.force(qh)
		codistribution = self.system.codistribution
		def residual(x):
			q1,v1,l = self.system.position(x), self.system.velocity(x), self.system.lag(x)
			cod = codistribution((q0+q1)/2)
			return np.hstack([
				q1 - q0 - h*self.system.average_velocity(u,x),
				v1 - v0 - h * (self.system.average_force(u,x) + np.dot(cod.T, l)),
				np.dot(cod, self.system.average_velocity(u,x)),
				])
		N = self.root_solver(residual)
		y = N.run(u)
		qnew, vnew, lnew = self.system.position(y), self.system.velocity(y), self.system.lag(y)
		return t+h, self.system.assemble(qnew,vnew,lnew)

class RKDAE(Scheme):
	"""
Partitioned Runge-Kutta for index 2 DAEs.
	"""

	root_solver = _rt.FSolve

	def __init__(self, tableau):
		super(RKDAE, self).__init__()
		self.tableau = tableau
		self.ts = self.get_ts()
		nb_stages = len(tableau) - 1
		self.QT = np.eye(nb_stages+1, nb_stages)

	def get_ts(self):
		return RungeKutta.time_vector(self.tableau)

	def dynamics(self, YZT):
		return np.dot(self.system.dynamics(YZT[:,:-1]), self.tableau[:,1:].T)

	def get_residual_function(self, t, u, h):
		T = t + self.ts*h
		y = self.system.state(u).copy()
		yc = y.reshape(-1,1) # "column" vector
		QT = self.QT

		def residual(YZ):
			YZT = np.vstack([YZ,T])
			dyn = self.dynamics(YZT)
			Y = self.system.state(YZ)
			r1 = Y - yc - h*dyn
			r2 = np.dot(self.system.constraint(YZT), QT)
			# unused final versions of z
			r3 = self.system.lag(YZ[:,-1])
			return [r1,r2,r3]

		return residual

	def step(self, t, u, h):
		s = self.tableau.shape[0]
		residual = self.get_residual_function(t, u, h)
		# pretty bad guess here:
		guess = np.column_stack([u.copy()]*s)
		N = self.root_solver(residual)
		residual(guess)
		full_result = N.run(guess) # all the values of Y,Z at all stages
		# we keep the last Z value:
		result = np.hstack([self.system.state(full_result[:,-1]), self.system.lag(full_result[:,-2])])
		return t+h, result

class MultiRKDAE(RKDAE):
	def dynamics(self, YZT):
		s = self.nb_stages
		dyn_dict = self.system.multi_dynamics(YZT[:,:-1])
		dyn = [np.dot(vec, RK_class.tableaux[s][:,1:].T) for RK_class, vec in dyn_dict.items()]
		return sum(dyn)


class Spark(MultiRKDAE):
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

The corresponding :class:`odelab.system.System` object must implement a *tensor* version the following methods:

* :meth:`odelab.system.System.state`
* :meth:`odelab.system.System.multi_dynamics`
* :meth:`odelab.system.System.constraint`
* :meth:`odelab.system.System.lag`

By vector, it is meant that methods must accept vector arguments, i.e., accept a nxs matrix as input parameter.

References:

.. [jay03] \L. Jay - *Solution of index 2 implicit differential-algebraic equations by Lobatto Runge-Kutta methods.* BIT 43, 1, 93-106 (2003). :doi:`10.1023/A:1023696822355`
	"""

	def __init__(self, nb_stages):
		self.nb_stages = nb_stages
		super(Spark, self).__init__(LobattoIIIA.tableaux[nb_stages])
		self.QT = self.compute_mean_stage_constraint().T


	def compute_mean_stage_constraint(self):
		"""
		Compute the s x (s+1) matrix defined in [jay03]_.
		"""
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
