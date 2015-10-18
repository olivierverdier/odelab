# -*- coding: UTF-8 -*-
from __future__ import division



from odelab.scheme import *
from odelab.scheme.classic import *
from odelab.scheme.exponential import *

from odelab.system import *
from odelab.solver import *
import odelab.newton as rt

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

class Harness_Circle(object):
	def setUp(self):
		self.f = rotational
		self.set_scheme()
		self.s = SingleStepSolver(self.scheme, DummySystem(self.f))
		self.scheme.h = .01
		self.s.initialize(u0 = array([1.,0.]))
		self.s.run(time = 10.)

	def test_plot_2D(self):
		plt.clf()
		a = self.s.plot(1,time_component=0).axis
		self.assertEqual(a.get_xlabel(), 'x')
		self.s.plot2D()
		for l in a.get_lines():
			d = l.get_data()
			radii = np.abs(np.sqrt(d[0]**2+d[1]**2) - 1)
			assert np.all(radii < .2) # should roughly be a circle

	def test_plot(self):
		a = self.s.plot(plot_exact=False).axis
		self.assertEqual(a.get_xlabel(), 'time')
		self.s.plot(plot_exact=True)
		tmp = tempfile.gettempdir()
		path = os.path.join(tmp, 'test_fig.pdf')
		print path
		plotter = Plotter(self.s)
		plotter.savefig(path)
		quick_setup(plotter, components=['output', 0], plot_exact=False)
		plotter.savefig(path)
		self.assertEqual(len(plotter.axis.lines), 2)
		quick_setup(plotter, components='output',)
		plotter.savefig(path)
		quick_setup(plotter, components=['output', 0], plot_exact=True)
		plotter.savefig(path)
		self.assertEqual(len(plotter.axis.lines), 4)
		# the following should be tested:
		self.s.plot(components=['output'], error=True)
		self.s.plot_function('output')
		quick_setup(plotter, components=[observer], plot_exact=True)
		plotter.savefig(path)

	def test_resolution(self, max_res=10):
		p = self.s.plot(plot_exact=False)
		p.max_plot_res = max_res
		p.plot()
		a = p.axis
		nb_points = a.get_lines()[-1].get_data()
		self.assertLessEqual(len(nb_points), max_res)

	def test_t0(self):
		t0 = 1
		p = self.s.plot(t0=t0)
		p.plot()
		a = p.axis
		data = a.get_lines()[-1].get_data()
		## bound = a.get_xbound()
		bound = data[0][0]
		self.assertAlmostEqual(bound,t0, places=0)

	def test_time(self):
		time = 2.
		p = self.s.plot(time=time)
		a = p.axis
		data = a.get_lines()[-1].get_data()
		self.assertLessEqual(data[0][-1],time)

class Test_Circle_EEuler(Harness_Circle, unittest.TestCase):
	def set_scheme(self):
		self.scheme = ExplicitEuler()

class Test_Circle_IEuler(Harness_Circle, unittest.TestCase):
	def set_scheme(self):
		self.scheme = ImplicitEuler()


class Test_Circle_RK34(Harness_Circle, unittest.TestCase):
	def set_scheme(self):
		self.scheme = RungeKutta34()
