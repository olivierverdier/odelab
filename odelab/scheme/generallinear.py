#!/usr/bin/env python
# coding: utf-8
from __future__ import division

from odelab.scheme import Scheme

import numpy as np
from numpy import array, sqrt

class MultistepInitializationError(ValueError):
	"""
Raised when not enough initial steps are available to start the multistep scheme.
	"""

class GeneralLinear(Scheme):
	pass

class ExplicitGeneralLinear(GeneralLinear):
	"""
	Scheme for explicit general linear methods.
	"""
	scaled_input = True # whether to have the input as (y_0, hf(y_1), hf(y_2)...)

	@property
	def tail_length(self):
		return len(self.tableau[1])

	def initialize(self,events):
		super(ExplicitGeneralLinear, self).initialize(events)
		tail_length = self.tail_length
		tail = np.array(events[:,-self.tail_length:])
		if len(tail.T) < tail_length:
			raise MultistepInitializationError('Only {0}/{1} past values given to initialize this multistep scheme'.format(len(tail.T),tail_length))
		if self.scaled_input:
			for i in range(tail.shape[1]-1):
				tail[:-1,i] = self.h*self.system.f(tail[-1,i], tail[:-1,i])
		self.tail = np.array(tail[:-1,::-1])


	def step(self, t, u, h):
		f = self.system
		ua, vb = self.tableau
		nb_stages = len(ua)
		nb_steps = len(vb)
		Y = np.zeros([len(u), nb_stages+nb_steps], dtype=u.dtype)
		Y[:,-nb_steps:] = self.tail
		newtail = np.zeros_like(self.tail) # alternative: work directly on self.tail
		for s in range(nb_stages):
			uas = ua[s]
			for j, coeff in enumerate(uas[1:]):
				if coeff is not None:
					Y[:,s] += coeff * Y[:,j]
			Y[:,s] = h*f(t+uas[0]*h, Y[:,s]) # possible since the method is explicit
		for r in range(nb_steps):
			vbr = vb[r]
			for j, coeff in enumerate(vbr):
				if coeff is not None:
					newtail[:,r] += coeff * Y[:,j]
		self.tail = newtail
		return t + h, self.tail[:,0]


class Heun(ExplicitGeneralLinear):
	tableau = ([[0,None,None,None,1],
		[1/3,1/3,None,None,1],
		[2/3, 0,2/3,None,1]],
		[[1/4,0,3/4,1]]
	)

class Kutta(ExplicitGeneralLinear):
	pass

class Kutta4(Kutta):
	tableau = ([	[0, None, None, None, None,1],
			[1/2, 1/2, None, None, None,1],
			[1/2, None, 1/2, None, None,1],
			[1., None, None, 1, None,1]],
				[[1/6, 1/3, 1/3, 1/6, 1]])

class Kutta38(Kutta):
	tableau = ([[0,None,None,None,None,1],
		[1/3,1/3,None,None,None,1],
		[2/3,-1/3, 1.,None,None,1],
		[1, 1, -1, 1, None,1]],
		[[1/8,3/8,3/8,1/8,1]])

class AdamsBashforth(ExplicitGeneralLinear):
	pass

class AdamsBashforth1(AdamsBashforth):
	tableau = ([[0, None, 1]], [[1, 1]])

class AdamsBashforth2(AdamsBashforth):
	tableau = ([[0, None, 1, 3/2,-1/2]],
		[	[None,1, 3/2,-1/2],
			[1, None, None, None],
			[None,None,1,None]
		])

class AdamsBashforth2e(AdamsBashforth):
	tableau = ([	[0, None, None, 1, None],
			[1., 3/2, None, 1, -1/2]
		],
		[	[3/2, None, 1, -1/2],
			[1, None, None, None]
		])

class Butcher(ExplicitGeneralLinear):
	scaled_input = False

class Butcher1(Butcher):
	tableau = ([[0, None, None, 1, 0],
			[1., 2, None, 0, 1]],
			[[5/4, 1/4, 1/2, 1/2],
			[3/4, -1/4, 1/2, 1/2]])

class Butcher3(Butcher):
	tableau = ([[0, None, None, 1, 0],
		[1, None, None, 0, 1]],
		[[-3/8, -3/8,-3/4,7/4],
		[-7/8,9/8,-3/4,7/4]])
