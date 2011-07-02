## -*- coding: UTF-8 -* -*- coding: UTF-8 -*-

from __future__ import division

import numpy as np

from odelab.scheme import Scheme

from odelab.scheme.rungekutta import *

import odelab.newton as _rt

class McLachlan(Scheme):
	ur"""
Solver for the Lagrange-d'Alembert (LDA) equations using the
algorithm given by equation (4.18) in [mclachlan06]_.

The scheme will work for any system as long as the necessary methods are implemented.
The adapted scheme reads as follows:

.. math::
	q_{1/2} &= q_0 + \frac{h}{2}H_p(q_0,p_0) \\
	p_1 &= p_0 + h\frac{H_q(q_{1/2},p_0) + H_q(q_{1/2},p_1)}{2} + h ∑_i λ_i α^i(q_{1/2})\\
	q_1 &= q_{1/2} + \frac{h}{2}H_p(q_1,p_1)\\
	&\bracket{θ(q_1)}{H_p(q_1,p_1)} = 0



The :class:`odelab.system.System` object must implement:

* :meth:`odelab.system.System.velocity`
* :meth:`odelab.system.System.momentum`
* :meth:`odelab.system.System.position`
* :meth:`odelab.system.System.lag`
* :meth:`odelab.system.System.force`
* :meth:`odelab.system.System.codistribution`


:References:

.. [mclachlan06] \R. McLachlan and M. Perlmutter, *Integrators for Nonholonomic Mechanical Systems*, J. Nonlinear Sci., **16** 283-328, (2006) :doi:`/10.1007/s00332-005-0698-1>`
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

	root_solver = _rt.FSolve

	def delta(self, t, u0, h):
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
		N = self.root_solver(residual)
		du = N.run(np.zeros_like(u0))
		return t+h, du

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
	root_solver = _rt.Newton

	def step(self, t, u0, h):
		v0 = self.system.velocity(u0)
		q1 = self.system.position(u0) + h*v0
		p0 = self.system.momentum(u0)
		force = self.system.force(q1)
		def residual(vl1):
			u1 = np.hstack([q1, vl1])
			v1 = self.system.velocity(u1)
			p1 = self.system.momentum(u1)
			l = self.system.lag(u1)
			codistribution = self.system.codistribution(q1)
			return np.hstack([p1 - p0 - h*(force + np.dot(codistribution.T, l)), np.dot(codistribution, v1)])
		N = self.root_solver(residual)
		l0 = self.system.lag(u0)
		vl1 = N.run(np.hstack([v0,l0]))
		u1 = np.hstack([q1,vl1])
		return t+h, u1

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

class RKDAE(Scheme):
	"""
Partitioned Runge-Kutta for index 2 DAEs.
	"""

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

	def get_residual(self, t, u, h):
		T = t + self.ts*h
		QT = self.QT

		def residual(DYZ):
			YZ = u.reshape(-1,1) + DYZ
			YZT = np.vstack([YZ,T])
			dyn = self.dynamics(YZT)
			DY = self.system.state(DYZ)
			Y = self.system.state(YZ)
			r1 = DY - h*dyn
			r2 = np.dot(self.system.constraint(YZT), QT)
			# unused final versions of z
			r3 = self.system.lag(YZ[:,-1])
			return [r1,r2,r3]

		return residual

	def get_guess(self, t, u, h):
		s = self.tableau.shape[0]
		# pretty bad guess here:
		guess = np.column_stack([np.zeros_like(u)]*s)
		return guess

	def reconstruct(self, full_result):
		# we keep the last Z value:
		result = np.hstack([self.system.state(full_result[:,-1]), self.system.lag(full_result[:,-2])])
		return  result

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
