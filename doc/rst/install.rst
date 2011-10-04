.. _Installation:

Installation
************

Instructions
============

|project| itself may be installed from the `odelab github repository`_ by running:

.. code-block:: sh

    pip install -e git+https://github.com/olivierverdier/odelab#egg=odelab

|project| depends ``scipy``, ``numpy`` and ``matplotlib`` in order to function properly.

There are also packages that, although not strictly necessary, improve |project| significantly:

* `PyTables`_, a package for reading and writing files in the `HDF5 format`_. Without this package, you will not be able to automatically store the simulation results on disk, and the memory usage may by higher.
* `python-progressbar`_, an optional package to display progress bars. Mostly interesting for very long simulation, as it indicates an estimate of the time left before the simulation is finished.

    
The only package which might be problematic to install is ``PyTables``.

.. _PyTables: https://github.com/PyTables/PyTables
.. _python-progressbar: https://github.com/olivierverdier/python-progressbar
.. _Theano: https://github.com/Theano/Theano
.. _odelab github repository: https://github.com/olivierverdier/odelab

.. _HDF5 format: http://www.hdfgroup.org/HDF5/


Test the installation
=====================

You may try to run the test suite of |project|.
(Make sure that you have `nose`_ installed first.)
To run the tests, go to the root directory of |project| and run:

.. code-block:: sh

    nosetest tests

.. _nose: http://readthedocs.org/docs/nose/en/latest/
