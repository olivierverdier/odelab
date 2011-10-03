.. _Installation:

Installation
************

Instructions
============

Quick installation instructions.
|project| depends on other packages in order to function properly, apart from ``scipy``, ``numpy`` and ``matplotlib``:

* `PyTables`_, a package for reading and writing files in the `HDF5 format`_.
* `python-progressbar`_, an optional package to display progress bars

    
The only package which might be problematic to install is ``PyTables``.

.. _PyTables: https://github.com/PyTables/PyTables
.. _python-progressbar: https://github.com/olivierverdier/python-progressbar
.. _Theano: https://github.com/Theano/Theano

.. _HDF5 format: http://www.hdfgroup.org/HDF5/


Test the installation
=====================

You may try to run the test suite of |project|.
(Make sure that you have `nose`_ installed first.)
To run the tests, go to the root directory of |project| and run:

.. code-block:: sh

    nosetest tests

.. _nose: http://readthedocs.org/docs/nose/en/latest/
