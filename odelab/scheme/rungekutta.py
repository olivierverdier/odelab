# -*- coding: UTF-8 -*-
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

	def __init__(self, nb_stages=None):
		super(ExplicitGeneralLinear, self).__init__()
		if nb_stages is not None:
			self.nb_stages = nb_stages

	@property
	def tail_length(self):
		return len(self.tableaux[self.nb_stages][1])

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
		f = self.system.f
		ua, vb = self.tableaux[self.nb_stages]
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
	nb_stages = 3
	tableaux = {
	3: ([[0,None,None,None,1],
		[1/3,1/3,None,None,1],
		[2/3, 0,2/3,None,1]],
		[[1/4,0,3/4,1]]
	)
	}

class Kutta(ExplicitGeneralLinear):
	tableaux = {
	4: ([	[0, None, None, None, None,1],
			[1/2, 1/2, None, None, None,1],
			[1/2, None, 1/2, None, None,1],
			[1., None, None, 1, None,1]],
				[[1/6, 1/3, 1/3, 1/6, 1]]),

	38: ([[0,None,None,None,None,1],
		[1/3,1/3,None,None,None,1],
		[2/3,-1/3, 1.,None,None,1],
		[1, 1, -1, 1, None,1]],
		[[1/8,3/8,3/8,1/8,1]])
	}


class AdamsBashforth(ExplicitGeneralLinear):


	tableaux = {
	1: ([[0, None, 1]], [[1, 1]]),
	2: ([[0, None, 1, 3/2,-1/2]],
		[	[None,1, 3/2,-1/2],
			[1, None, None, None],
			[None,None,1,None]
		]),
	'2e': ([	[0, None, None, 1, None],
			[1., 3/2, None, 1, -1/2]
		],
		[	[3/2, None, 1, -1/2],
			[1, None, None, None]
		])

	}

class Butcher(ExplicitGeneralLinear):
	scaled_input = False

	tableaux = {
	1: ([[0, None, None, 1, 0],
			[1., 2, None, 0, 1]],
			[[5/4, 1/4, 1/2, 1/2],
			[3/4, -1/4, 1/2, 1/2]]),
	3: ([[0, None, None, 1, 0],
		[1, None, None, 0, 1]],
		[[-3/8, -3/8,-3/4,7/4],
		[-7/8,9/8,-3/4,7/4]])
	}
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

class RadauIIA(RungeKutta):
	ss = sqrt(6)
	tableaux = {
2: array([[1/3, 5/12, -1/12],[1., 3/4, 1/4],[1.,3/4,1/4]]),
3: array([	[(4-ss)/10, (88-7*ss)/360, (296 - 169*ss)/1800, (-2+3*ss)/225],
			[(4+ss)/10, (296+169*ss)/1800, (88+7*ss)/360, (-2-3*ss)/225],
			[1., (16-ss)/36, (16+ss)/36, 1/9],
			[1., (16-ss)/36, (16+ss)/36, 1/9]])
	}
