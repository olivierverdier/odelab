# -*- coding: utf-8 -*-
from __future__ import division

from odelab.order import *
from odelab.solver import *
from odelab.scheme.exponential import *
from odelab.system import *

class Order_Burgers(object):
	def test_run(self):
		B = BurgersComplex(viscosity=.03)
		size = B.size
		u0 = .5 - np.abs(B.points)
## 		u0 = np.cos(B.points*(2*np.pi))*(1+np.sin(B.points*(2*np.pi)))
		info = {'u0':u0, 't0':0., 'time':.5, }

		solvers = [MultiStepSolver(scheme_class(), B) for scheme_class in [LawsonEuler, RKMK4T, HochOst4, GenLawson45,ABNorset4,Lawson4,ABLawson4][3:4]]
		self.o = OrderFarm(solvers, B, info)
		self.o.initialize()
		self.o.run(np.linspace(2,3,5))
		print(self.o.orders())
		self.o.plot()

