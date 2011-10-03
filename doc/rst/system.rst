Systems
********

In |project|, a *system* generalizes the notion of a right-hand side.
The rationale behind having a class to model it is that there is often a particular *structure* present in the right hand side.

For example, the system might be a mechanical system, in which case we have to describe which part of the system is a force, and which part is a velocity (or a momentum).
A system may also be compose of a fast and a slow component.
It may be a *partitioned* system, i.e., the right-hand side may be different for different components.
It may be a sum of a constant, linear part and a non-linear part, in which case some exponential integrators may be used.

It is one of the long-term goals of this project to include all the common dynamical systems used in textbooks on differential equations, such as the van der Pol, Lotka-Volterra, Kepler systems.
Some of those are already available as a subclass of :class:`odelab.system.System`.


Classical Systems
=================

A collection of classical systems.

.. automodule:: odelab.system.classic
	:members:

