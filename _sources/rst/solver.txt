Solvers
********

The :class:`Solver` class takes care of

* initializing the problem
* running the simulation
* storing the results efficiently

Using the :class:`Solver` class
===============================

Initialize and Run
------------------

You would generally start by picking a system ``system`` and a numerical scheme ``scheme`` and initialize the solver object as::

    solver = Solver(scheme=scheme, system=system)

The most important methods are then :meth:`initialize` and :meth:`run`.

The initialization method is run as follows, for a given initial data ``u0``::

    solver.initialize(u0=u0)

The next method to use is :meth:`run`, which actually runs the simulation.
The required parameter is ``time`` which indicates the time span of the simulation::

    solver.run(time=1.)


Storage
=======

One of the most valuable features of |project| is the data storage.
In many cases, either because the simulation is run for a very long time, or because the dimension of the phase space is very large, the produced data may be significant.
In those cases, you do not want the whole data to be stored in memory, especially since the past data is hardly of any interest to the numerical scheme (which only needs a few steps back to move forward).

Convenience Methods
-------------------

The data may easily be accessed using some convenience methods.

The methods :meth:`~odelab.solver.Solver.initial` and :meth:`~odelab.solver.Solver.final` give access to the initial and final state of the system.

The method :meth:`~odelab.solver.Solver.get_events` may be used to extract some of the produced results, on a given time span, and with a given sampling rate.

Plotting the Data
-----------------

Components
----------

One may quickly plot the results using the method :meth:`~odelab.solver.Solver.plot`.
The simplest use of that method is to plot all the coordinates on the whole time span, with::

    solver.plot()

One may restrict the plotted coordinates using by given them in a list.
For instance, to only plot the first and third coordinates::

    solver.plot([0,2])

Generalized Components
----------------------

Note that a method of the corresponding :class:`System` object may also be used as components (what we call here *generalized component*).
For example, if the system object implements a method ``energy`` which computes the energy of a given state, as::

    def energy(self, t,u):
        return ...

then it is possible to plot the energy using::

    solver.plot('energy')

It is possible to combine it with other components, or other :class:`System` methods, by gathering them in a list::

    solver.plot([0,'energy']) # plot the first coordinate, and the energy

2D Plot
-------

It is also possible to plot one coordinate against another.
Suppose that coordinates 0 and 1 are positions in a plane.
The trajectory is conveniently plotted with::

    solver.plot(1, time_component=0)

The method :meth:`~odelab.solver.Solver.plot2D` is an alias to the call above.
One may thus call::

    solver.plot2D(0,1) # plot coordinate 1 against coordinate 0

.. topic:: Example: Solving the van der Pol equation

    Here is an example of solving the van der Pol equations with a Runge-Kutta method and then plotting the result with ``plot2D``:

    .. plot:: code/vanderpol.py
        :include-source:


Retrieving the data
-------------------

Since the data is stored in a file, the file must be opened when retrieving the data, and closed after that.
This is achieved using the :meth:`open_store` context manager::

    with solver.open_store() as events:
        first_component = events[0]

Format of the data
------------------

The data is stored in a matrix of size :math:`N \times T`, where :math:`N` is the size of the phase space, and :math:`T` is the number of time points.
In order to obtain the first component of the simulated vector, you would do as follows::

    with solver.open_store() as events:
        events[0] # first component
        events[0,5] # first component at time offset 5
        events[:,0] # the initial condition

.. note::
    Note that the initial and final conditions may be obtained with :meth:`initial` and :meth:`final` respectively.

.. _Storage:


File and simulation name
------------------------

|project| *always* stores the result in a file.
For convenience, if no file name is specified, they are stored inside a temporary file.

If you want to create a file in which to store the data, you may do so by using the ``path`` parameter when creating the :class:`Solver` object::

    solver = Solver(scheme=scheme, system=system, path='...')

The given path must be either a path to a new file to be created by |project|, or to an existing file *of type HDF5*.
It could, for instance, point to the file storage of an earlier simulation.

Each simulation has a name, which is also automatically created if none is specified.
The name is given at the initialization::

    solver.initialize(..., name='my_simulation')

The corresponding simulation may then be loaded at any later time using::

    solver = load_solver(path, name)



API
===

.. automodule:: odelab.solver
	:members:
