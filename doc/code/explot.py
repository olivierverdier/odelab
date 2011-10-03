from odelab import Solver, System
from odelab.scheme.classic import ExplicitEuler

def f(t,u):
	return -u

s = Solver(scheme=ExplicitEuler(.1), system=System(f))
s.initialize(u0=1.)
s.run(1.)

s.plot()
