#!/usr/bin/env python
# coding: utf-8
from __future__ import division

from . import Scheme
import numpy as np
from numpy import array, sqrt

class RungeKutta(Scheme):
	"""
	Collection of classes containing the coefficients of various Runge-Kutta methods.

	:Attributes:
		tableaux : dictionary
			dictionary containing a Butcher tableau for every available number of stages.
	"""
	@classmethod
	def time_vector(cls, tableau):
		return tableau[:,0]

class RKDAE(RungeKutta):
	"""
Partitioned Runge-Kutta for index 2 DAEs.
	"""

	def __init__(self, *args, **kwargs):
		super(RKDAE, self).__init__(*args, **kwargs)
		tableau = kwargs.get('tableau')
		if tableau is not None:
			self.tableau = tableau
		self.ts = self.get_ts()
		self.nb_stages = len(self.tableau) - 1
		self.QT = np.eye(self.nb_stages+1, self.nb_stages)

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

	def reconstruct(self, full_result,t,u0,h):
		# we keep the last Z value:
		result = np.hstack([self.system.state(full_result[:,-1]), self.system.lag(full_result[:,-2])])
		return  h, result

class MultiRKDAE(RKDAE):
	def dynamics(self, YZT):
		s = self.nb_stages
		dyn_dict = self.system.multi_dynamics(YZT[:,:-1])
		dyn = [np.dot(vec, RK_class.tableaux[s][:,1:].T) for RK_class, vec in dyn_dict.items()]
		return sum(dyn)


class Spark(MultiRKDAE):
	r"""
Solver for semi-explicit index 2 DAEs by Lobatto III RK methods
 as presented in [Ja03]_.

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

.. [Ja03] \L. Jay - *Solution of index 2 implicit differential-algebraic equations by Lobatto Runge-Kutta methods.* BIT 43, 1, 93-106 (2003). :doi:`10.1023/A:1023696822355`
	"""

	def __init__(self, *args, **kwargs):
		super(Spark, self).__init__(*args, **kwargs)
		self.QT = self.compute_mean_stage_constraint().T


	def compute_mean_stage_constraint(self):
		"""
		Compute the s x (s+1) matrix defined in [Ja03]_.
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


class ImplicitEuler(RungeKutta):
	tableaux = {
		1: array([[1,1],[1,1]])
			}

class LDIRK343(RungeKutta):
	gamma = 0.43586652150846206
	b1 = -1/4 + gamma*(4 - 3/2*gamma)
	b2 = 5/4 + gamma*(-5 + 3/2*gamma)
	tableaux = {
	3: array([[gamma,gamma, 0, 0],
		[(1+gamma)/2,(1.-gamma)/2, gamma, 0],
		[1,b1, b2, gamma],
		[1.,b1, b2, gamma]],
		),
			}

class LobattoIIIA(RungeKutta):

	sf = sqrt(5)
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
	sf = sqrt(5)
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
	sf = sqrt(5)
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

class Spark2(Spark):
	tableau = LobattoIIIA.tableaux[2]

class Spark3(Spark):
	tableau = LobattoIIIA.tableaux[3]

class Spark4(Spark):
	tableau = LobattoIIIA.tableaux[4]

class RadauIIA(RungeKutta):
	ss = sqrt(6)
	tableaux = {
2: array([[1/3, 5/12, -1/12],[1., 3/4, 1/4],[1.,3/4,1/4]]),
3: array([	[(4-ss)/10, (88-7*ss)/360, (296 - 169*ss)/1800, (-2+3*ss)/225],
			[(4+ss)/10, (296+169*ss)/1800, (88+7*ss)/360, (-2-3*ss)/225],
			[1., (16-ss)/36, (16+ss)/36, 1/9],
			[1., (16-ss)/36, (16+ss)/36, 1/9]])
	}
