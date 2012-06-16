# ODE Solvers in Python

Numerical simulation of ODEs in Python ([full documentation][doc] available).

## Features

 * simulation of differential equations with various solvers
 * emphasis on long-time simulation, and efficient storage of data
 * symplectic solvers for mechanical systems
 * Generic Runge-Kutta solvers
 * Exponential solvers



## Example

![van der Pol example](http://olivierverdier.github.com/odelab/_images/vanderpol.png)

That example can be reproduced with:

```python
from odelab import Solver
from odelab.scheme.classic import RungeKutta4
from odelab.system.classic import VanderPol

s = Solver(scheme=RungeKutta4(.1), system=VanderPol())
s.initialize([2.7,0], name='2.7')
s.run(10.)
s.plot2D()
```

Read the full [full documentation][doc].

[doc]: http://olivierverdier.github.com/odelab/

