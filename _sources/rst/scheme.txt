.. _scheme_chapter:

Scheme
**********************************

The :class:`Scheme` class may be used to easily create new schemes.

Let us examine the code to create the Explicit Euler scheme::

    from odelab import Scheme

    class ExplicitEuler(Scheme):
        def delta(self, t, u0, h):
            return h*self.system.f(t,u0)

An object of the class :class:`Scheme` has access to the attribute :attr:`system`, which gives access to the system at hand.

If the method is implicit, all you have to do is to implement the :meth:`get_residual` method instead.
Here is an example showing how to implement the Implicit Euler scheme::

    class ImplicitEuler(Scheme):
        def get_residual(self, t, u0, h):
            def residual(du):
                return du - h*self.system.f(t+h, u0+du)
            return residual

Main Methods
============

The :class:`odelab.scheme.Scheme` class provides several methods to assist the creation of numerical schemes.

Explicit Schemes
----------------

The method :meth:`~odelab.scheme.Scheme.delta` may be used to directly provide the difference :math:`u_1 - u_0`.
This is typically useful for explicit scheme.
See for instance :class:`odelab.scheme.classic.RungeKutta4`, or :class:`odelab.scheme.classic.ExplicitEuler`.

Implicit Schemes
----------------

General Residual
^^^^^^^^^^^^^^^^

For implicit schemes, the method :meth:`~odelab.scheme.Scheme.get_residual` simply returns the residual of a function which root is the difference :math:`u_1 - u_0`.
In other words, the residual is a function :math:`F` such that

.. math::
    F(u_1 - u_0) = 0

Custom Residual
^^^^^^^^^^^^^^^

Sometimes, the residual's root is not the difference :math:`u_1 - u_0`, and |project| supports that as well.
The best way to understand how it works is to look at the implementation of :meth:`~odelab.scheme.Scheme.delta`, which uses :meth:`~odelab.scheme.Scheme.get_residual`.

In short, the method ``delta`` runs a nonlinear solver on the residual provided by ``get_residual``, using a guess provided by :meth:`~odelab.scheme.Scheme.get_guess`.
The root is then passed to the method :meth:`~odelab.scheme.Scheme.reconstruct`, which job is to construct the difference :math:`u_1 - u_0` from the root of the residual.

This flexibility is useful, for instance, if only one part of the method is implicit.

Time Adaptivity
===============

The methods :meth:`~odelab.scheme.Scheme.delta`, :meth:`~odelab.scheme.Scheme.get_residual` or :meth:`~odelab.scheme.Scheme.step` have ``h`` as a parameter, but it is just a convenience.
The time step parameter is in fact stored by the :class:`~odelab.scheme.Scheme` object itself.
A time adaptive scheme may thus change the next time step, or even implement advanced time-step control.

An example of time-adaptive scheme is given by :class:`~odelab.scheme.classic.RungeKutta34`.

Initialization and External Solvers
===================================

It is the user's responsibility to initialize the scheme object appropriately.
Usually, the only necessary argument is the time-step::

    ie = ImplicitEuler(h=.1)

Note that the solver calls the :meth:`~odelab.scheme.Scheme.initialize` method before running the simulation.
At this stage, the solver object has been initialized, and in particular has a initial condition and a :class:`~odelab.system.System` object.
This is especially useful for using external numerical schemes, as those often require the right hand side and initial conditions to be known at initialization.
An example of scheme using an external numerical scheme is given by :class:`~odelab.scheme.ode15s`, which uses SciPy's function ``odeint``.

Family of Numerical Schemes
===========================

|project| comes with a variety of numerical schemes to be used out of the box.
For instance:

 * generic Runge-Kutta methods
 * generic linear multi-step methods
 * generic explicit general linear methods
 * generic exponential methods

”Classical“ Numerical Scheme
============================

A collection of classical numerical scheme described in all the basic textbooks on differential equations.

.. automodule:: odelab.scheme.classic
    :members:


The :class:`Scheme` class
=========================

.. automodule:: odelab.scheme
    :members:


