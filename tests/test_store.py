#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import pytest

from odelab.store import SimpleStore, PyTableStore, Store
import numpy as np

@pytest.fixture(params=[SimpleStore, PyTableStore])
def store(request):
	store_class = request.param
	s = store_class()
	s.initialize(np.array([1., 0]), name='foo')
	return s

class TestStore(object):
	def test_open(self, store):
		store.append(np.array([1.,1]))
		store.append(np.array([1.,2]))
		assert len(store) == 2
		with store.open() as events:
			assert events.shape == (2,2)

	def test_append(self, store):
		store.append(np.array([1.,1]))
		assert len(store) == 1
		with store.open() as events:
			assert events.dtype == float

	def test_openappend(self, store):
		if isinstance(store, PyTableStore):
			pytest.skip()
		with store.open() as events:
			store.append(np.array([1.,1]))
		assert len(store) == 1
		with store.open() as events:
			assert events.dtype == float


class Test_Exceptions(unittest.TestCase):
	def setUp(self):
		if Store is SimpleStore:
			pytest.skip()
		self.s = PyTableStore()

	def test_raise_not_initialized(self):
		with self.assertRaises(PyTableStore.NotInitialized):
			len(self.s)
