#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import odelab.scheme.rungekutta as rk
from odelab.system import System
import numpy as np
from numpy import array

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

.. [jay06] Jay - *Specialized Runge-Kutta methods for index $ 2$ differential-algebraic equations.* Math. Comput. 75, No. 254, 641-654 (2006). :doi:`10.1090/S0025-5718-05-01809-0`
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

	@classmethod
	def exact(self, t, u0):
		if np.allclose(u0[:2], array([1.,1.])):
			return array([np.exp(t), np.exp(-2*t), np.exp(2*t)])
		raise ValueError("Exact solution not defined")

	def label(self, component):
		return ['y1','y2','z'][component]

