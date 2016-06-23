#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import pytest

from odelab.store import SimpleStore, PyTableStore, Store
import numpy as np


class Harness_Store(object):
	def test_open(self):
		self.s.append(np.array([1.,1]))
		self.s.append(np.array([1.,2]))
		self.assertEqual(len(self.s), 2)
		with self.s.open() as events:
			self.assertEqual(events.shape, (2,2))

	def test_append(self):
		self.s.append(np.array([1.,1]))
		self.assertEqual(len(self.s),1)
		with self.s.open() as events:
			self.assertEqual(events.dtype, float)

class Test_SimpleStore(Harness_Store, unittest.TestCase):
	def setUp(self):
		self.s = SimpleStore()
		self.s.initialize(np.array([1.,0]), name=None)

	def test_openappend(self):
		with self.s.open() as events:
			self.s.append(np.array([1.,1]))
		self.assertEqual(len(self.s),1)
		with self.s.open() as events:
			self.assertEqual(events.dtype, float)

class Test_PyTableStore(Harness_Store, unittest.TestCase):
	def setUp(self):
		if Store is SimpleStore:
			pytest.skip()
		self.s = PyTableStore()
		self.s.initialize(np.array([1., 0]), name='foo')

class Test_Exceptions(unittest.TestCase):
	def setUp(self):
		if Store is SimpleStore:
			pytest.skip()
		self.s = PyTableStore()

	def test_raise_not_initialized(self):
		with self.assertRaises(PyTableStore.NotInitialized):
			len(self.s)
