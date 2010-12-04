# -*- coding: UTF-8 -*-
"""
:mod:`System` -- Systems
============================

Examples of ODE systems.

Collection of subclasses of the class :class:`odelab.system.System`.
The interface needed depends on the solver. For example, exponential solvers will need a `linear` method,
dae solvers will need a `constraint`, or `multi-dynamics` method. See the documentation of the :doc:`scheme`
section for more information.

.. module :: system
	:synopsis: Examples of ODE systems.
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

"""
from __future__ import division

from base import *
from dae import *
from stochastic import *
from exponential import *
from nonholonomic import *







