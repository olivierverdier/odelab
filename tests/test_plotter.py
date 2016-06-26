# -*- coding: utf-8 -*-
from __future__ import division

import pytest

from odelab.scheme import *
from odelab.scheme.classic import *
from odelab.scheme.exponential import *

from odelab.system import *
from odelab.solver import *
import newton as rt

import tempfile
import os

import numpy as np
import numpy.testing as npt

import unittest

import matplotlib.pyplot as plt


Solver.catch_runtime = False

class DummySystem(System):
	def __init__(self, f):
		super(DummySystem,self).__init__(f)

	def label(self, component):
		return ['x', 'y'][component]

	def output(self, ut):
		return np.ones(ut.shape[1])

	def exact(self, t, e0):
		x,y,t0 = e0
		c,s = np.cos(t), np.sin(t)
		return np.vstack([c*x-s*y, s*x + c*y])

def observer(uts):
	return uts[0] + uts[1]

def rotational(t,u):
	"""
	Rotational vector field
	"""
	return array([-u[1], u[0]])

def quick_setup(plotter, **kwargs):
	for k,v in kwargs.items():
		setattr(plotter,k,v)

schemes = [ExplicitEuler()]

@pytest.fixture(scope='module', params=schemes, ids=repr)
def solver(request):
	f = rotational
	scheme = request.param
	solver = SingleStepSolver(scheme, DummySystem(f))
	scheme.h = .05
	solver.initialize(u0 = array([1.,0.]))
	solver.run(time = 3.)
	return solver

class TestCircle(object):

	def test_plot_2D(self, solver):
		plt.clf()
		a = solver.plot(1,time_component=0).axis
		assert a.get_xlabel() == 'x'
		solver.plot2D()
		for l in a.get_lines():
			d = l.get_data()
			radii = np.abs(np.sqrt(d[0]**2+d[1]**2) - 1)
			assert np.all(radii < .3) # should roughly be a circle

	def test_plot(self, solver):
		a = solver.plot(plot_exact=False).axis
		assert a.get_xlabel() == 'time'
		plotter = Plotter(solver)
		plotter.axis_plot()
		quick_setup(plotter, components=['output', 0], plot_exact=False)
		plt.gca().cla()
		plotter.axis_plot()
		assert len(plotter.axis.lines) == 2
		quick_setup(plotter, components='output',)
		plt.gca().cla()
		plotter.axis_plot()
		quick_setup(plotter, components=['output', 0], plot_exact=True)
		plt.gca().cla()
		plotter.axis_plot()
		assert len(plotter.axis.lines) == 4
		# the following should be tested:
		solver.plot(components=['output'], error=True)
		solver.plot_function('output')
		quick_setup(plotter, components=[observer], plot_exact=True)

	def test_resolution(self, solver, max_res=10):
		p = solver.plot(plot_exact=False)
		p.max_plot_res = max_res
		p.plot()
		a = p.axis
		nb_points = a.get_lines()[-1].get_data()
		assert len(nb_points) <= max_res

	def test_t0(self, solver):
		t0 = 1
		p = solver.plot(t0=t0)
		p.plot()
		a = p.axis
		data = a.get_lines()[-1].get_data()
		## bound = a.get_xbound()
		bound = data[0][0]
		assert round(bound - t0, 0) == 0

	def test_time(self, solver):
		time = 2.
		p = solver.plot(time=time)
		a = p.axis
		data = a.get_lines()[-1].get_data()
		assert data[0][-1] <= time

