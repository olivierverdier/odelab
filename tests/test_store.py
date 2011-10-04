#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division

from odelab.store import SimpleStore, PyTableStore, Store
import numpy as np

import nose.tools as nt
from nose.plugins.skip import SkipTest

class Harness_Store(object):
	def test_open(self):
		self.s.append(np.array([1.,1]))
		self.s.append(np.array([1.,2]))
		nt.assert_equal(len(self.s), 2)
		with self.s.open() as events:
			nt.assert_equal(events.shape, (2,2))

	def test_append(self):
		self.s.append(np.array([1.,1]))
		nt.assert_equal(len(self.s),1)
		with self.s.open() as events:
			nt.assert_equal(events.dtype, float)

class Test_SimpleStore(Harness_Store):
	def setUp(self):
		self.s = SimpleStore()
		self.s.initialize(np.array([1.,0]), name=None)

	def test_openappend(self):
		with self.s.open() as events:
			self.s.append(np.array([1.,1]))
		nt.assert_equal(len(self.s),1)
		with self.s.open() as events:
			nt.assert_equal(events.dtype, float)

class Test_PyTableStore(Harness_Store):
	def setUp(self):
		if Store is SimpleStore:
			raise SkipTest()
		self.s = PyTableStore()
		self.s.initialize(np.array([1., 0]), name='foo')
