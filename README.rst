ODE Solvers in Python
==============================

Numerical simulation of ODEs in Python (`full documentation`_ available).

Features
--------

 * simulation of differential equations with various solvers
 * emphasis on long-time simulation, and efficient storage of data
 * symplectic solvers for mechanical systems
 * Generic Runge-Kutta solvers
 * Exponential solvers



Example
--------

.. image:: http://olivierverdier.github.com/odelab/_images/vanderpol.png
    :alt: van der Pol example

That example can be reproduced with::

    from odelab import Solver
    from odelab.scheme.classic import RungeKutta4
    from odelab.system.classic import VanderPol

    s = Solver(scheme=RungeKutta4(.1), system=VanderPol())
    s.initialize([2.7,0], name='2.7')
    s.run(10.)
    s.plot2D()


Read the full `full documentation`_

.. _full documentation: http://olivierverdier.github.com/odelab/

