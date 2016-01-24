#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
	name = 'odelab',
	description = 'ODE Simulation Tools',
	author='Olivier Verdier',
	license = 'BSD2',
	keywords = ['Math', 'ode',],
	
	packages=['odelab', 'odelab.scheme', 'odelab.system', 'odelab.system.nonholonomic'],
	classifiers = [
	'Development Status :: 4 - Beta',
	'Intended Audience :: Science/Research',
	'License :: OSI Approved :: BSD License',
	'Operating System :: OS Independent',
	'Programming Language :: Python',
	'Topic :: Scientific/Engineering :: Mathematics',
	],
	)
