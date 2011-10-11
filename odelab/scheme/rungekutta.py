#!/usr/bin/env python
# coding: UTF-8
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
