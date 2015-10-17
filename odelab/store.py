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
import numpy as np

class SimpleStore(object):
	def __init__(self, path=None):
		self.info = {}

	def __getitem__(self, key):
		return self.info[key]

	def __setitem__(self, key, value):
		self.info[key] = value

	def initialize(self, event0, name):
		self.name = name
		self.events = []

	def __len__(self):
		return len(self.events)

	def append(self, event):
		self.events.append(event)

	@contextmanager
	def open(self, name=None, write=False):
		events = np.array(self.events).T
		yield events

	def get_nb_stage(self, events):
		return len(events.T)


class PyTableStore(SimpleStore):
	class AlreadyInitialized(Exception):
		"""
		Raised when a solver is already initialized.
		"""

	class NotInitialized(Exception):
		"""
		Raised when store was not properly initialized.
		"""

	def __init__(self, path=None):
		if path is None: # file does not exist
			import tempfile
			f = tempfile.NamedTemporaryFile(delete=True)
			self.path = f.name
		else:
			self.path = path

		# compression algorithm
		self.compression = tables.Filters(complevel=1, complib='zlib', fletcher32=True)

	def append(self, event):
		with self.open(write=True) as events:
			events.append(event.reshape(-1,1))

	def getitem(self, events, key):
		return events.attrs[key]

	def __getitem__(self, key):
		with self.open(write=True) as events:
			return self.getitem(events, key)

	def setitem(self, events, key, value):
		try:
			events.attrs[key] = value
		except TypeError as e: # TODO: test this
			warnings.warn(str(e))

	def __setitem__(self, key, value):
		with self.open(write=True) as events:
			return self.setitem(events, key, value)

	def initialize(self, event0, name):
		self.name = name
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

	def __len__(self):
		with self.open() as events:
			size = events.nrows
		return size

	def get_nb_stage(self, events):
		return len(events)

	@contextmanager
	def open(self, write=False):
		mode = ['r','a'][write]
		try:
			with tables.openFile(self.path, mode) as f:
				node = f.getNode('/'+self.name)
				yield node
		except IOError:
			raise self.NotInitialized('This store has not been initialized')

try:
	import tables
except ImportError:
	warnings.warn('PyTables is not available; falling back to a simpler store')
	Store = SimpleStore
else:
	Store = PyTableStore
## Store = SimpleStore
