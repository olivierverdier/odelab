#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import os

import tables

import time
import datetime
from datetime import timedelta


from pprint import PrettyPrinter

pprinter = PrettyPrinter()

class Experiment(object):
	def __init__(self, parameters, store_prefix=None):
		self.parameters = parameters
		if store_prefix is None:
			store_prefix = self.default_prefix()
		self.store_prefix = store_prefix

	def default_prefix(self):
		"""
Default prefix from the command line.
		"""
		import sys
		try:
			prefix = sys.argv[1]
		except IndexError:
			prefix = ''
		return prefix

	def get_system(self):
		sys_class = self.parameters['system']
		sys = sys_class(**self.parameters.get('system_params', {}))
		return sys

	def get_solver(self, file=None):
		parameters = self.parameters
		sys = self.get_system()
		scheme_class = parameters['scheme']
		scheme = scheme_class(**parameters.get('scheme_params', {}))
		solver_class = parameters['solver']
		if file is None:
			file = tables.openFile(self.get_path(), mode='a')
		solver = solver_class(scheme, sys, file=file, name=parameters['name'])
		return solver

	def run(self, save=True):
		parameters = self.parameters
		self.solver = self.get_solver()
		self.solver.initialize(**parameters['initialize'])
		print('-'*80)
		print(self.get_path())
		print(parameters['name'])
		pprinter.pprint(parameters)
		start_time = time.time()
		try:
			self.solver.run()
		except Exception as e:
			print(e)
		finally:
			end_time = time.time()
			self.duration = end_time - start_time
			print(str(timedelta(seconds=self.duration)))
			self.timestamp = datetime.datetime.now()
			if save:
				self.save()

	def get_path(self):
		path = os.path.join(self.store_prefix, self.parameters['family'])
		return path

	def save(self):
		parameters = self.parameters
		info = {}
		info['timestamp'] = self.timestamp
		info['duration'] = self.duration
		info['params'] = parameters
		self.solver.events.attrs['info'] = info

	@classmethod
	def load(self, path, name):
		hdfile = tables.openFile(path, mode='a')
		events = hdfile.getNode('/'+name)
		info = events.attrs['info']

		params = info['params']
		experiment = self(params, store_prefix='')
		solver = experiment.get_solver(file=hdfile)
		solver.load_data()
		return solver


