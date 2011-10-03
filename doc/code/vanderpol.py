from odelab import Solver
from odelab.scheme.classic import RungeKutta4
from odelab.system.classic import VanderPol

s = Solver(scheme=RungeKutta4(.1), system=VanderPol())
s.initialize([2.7,0], name='2.7')
s.run(10.)
s.plot2D()

s.initialize([1.3,0], name='1.3')
s.run(10.)
s.plot2D()
