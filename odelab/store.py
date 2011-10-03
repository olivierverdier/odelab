# -*- coding: UTF-8 -*-
"""
:mod:`Store` -- Store
============================

Stores data produced by the solver, if PyTables is available.

.. module :: store
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

"""

from contextlib import contextmanager
import warnings

class SimpleStore(object):
	pass

class PyTableStore(SimpleStore):
	class AlreadyInitialized(Exception):
		"""
		Raised when a solver is already initialized.
		"""

	def __init__(self, path):
		if path is None: # file does not exist
			import tempfile
			f = tempfile.NamedTemporaryFile(delete=True)
			self.path = f.name
		else:
			self.path = path

		# compression algorithm
		self.compression = tables.Filters(complevel=1, complib='zlib', fletcher32=True)

	def initialize(self, event0, name):
		with tables.openFile(self.path, 'a') as store:
			# create a new extensible array node
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				try:
					events = store.createEArray(
						where=store.root,
						name=name,
						atom=tables.Atom.from_dtype(event0.dtype),
						shape=(len(event0),0),
						filters=self.compression)
				except tables.NodeError:
					raise self.AlreadyInitialized('Results with the name "{0}" already exist in that store.'.format(name))

	@contextmanager
	def open(self, name, write=False):
		mode = ['r','a'][write]
		with tables.openFile(self.path, mode) as f:
			node = f.getNode('/'+name)
			yield node

import tables

Store = PyTableStore

