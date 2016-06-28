import pytest
import numpy.testing as npt

from odelab.scheme.rungekutta import *
from odelab.scheme import *
from odelab.scheme.classic import *
# from odelab.scheme.generallinear import *
# from odelab.scheme.exponential import *

from odelab import *

def f(t,u):
	return t*np.ones_like(u)
def const_f(c,t,u):
	return c*np.ones_like(u)
from functools import partial
const_r = partial(const_f, 1.)
const_c = partial(const_f, 1.j)
def time_f(t,u):
	return t

schemes = [
		ExplicitEuler(h=.1),
		ImplicitEuler(h=.1),
		ExplicitTrapezoidal(h=.1),
		RungeKutta4(h=.1),
		ImplicitMidPoint(h=.1),
		RungeKutta34(h=.1),
		ode15s(h=.1),
]

@pytest.fixture(params=schemes, ids=repr)
def scheme(request):
	return request.param

class TestLinQuad():
	def test_quadratic(self, scheme):
		sys = System(time_f)
		solver = Solver(system=sys, scheme=scheme)
		solver.initialize(u0=1., )
		solver.run(time=1.,)
		# u'(t) = t; u(0) = u0; => u(t) == u0 + t**2/2
		npt.assert_array_almost_equal(solver.final(), np.array([3/2,1.]), decimal=1)

	def check_const(self, f, u0, expected, scheme):
		"""should solve the f=c exactly"""
		sys = System(f)
		solver = Solver(system=sys, scheme=scheme)
		solver.initialize(u0=u0, )
		solver.run(time=1.,)
		expected_event = np.hstack([expected, 1.])
		npt.assert_almost_equal(solver.final(), expected_event, 1)

	def test_real_const(self, scheme):
		self.check_const(const_r, 1., 2., scheme)

	@pytest.mark.skip('Current nonlinear solver does not work with the complex type.')
	def test_complex_const(self, scheme):
		self.check_const(const_c, 1.+0j, 1.+1.j, scheme)

	def test_repr(self, scheme):
		expected = '<Solver: {0}'.format(repr(scheme))
		solver = Solver(scheme=scheme, system=System(f))
		r = repr(solver)
		assert r.startswith(expected)
		# if solver.init_scheme is not None:
		# 	self.assertRegex(r, repr(self.solver.init_scheme))


# class Test_AB(Harness_Solver, unittest.TestCase):
# 	def setup_solver(self):
# 		multi_scheme = AdamsBashforth2(.1)
# 		self.solver = Solver(multi_scheme, System(f), init_scheme=ExplicitEuler(h=.1))



# class Test_LawsonEuler(Harness_Solver_NoComplex, unittest.TestCase):
# 	def set_system(self, f):
# 		self.solver.system = NoLinear(f,self.dim)
# 	def setup_solver(self):
# 		self.solver = Solver(LawsonEuler(h=.1), NoLinear(f,self.dim))
