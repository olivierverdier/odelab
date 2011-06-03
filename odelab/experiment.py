#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import os

import shelve
import time
import datetime
from datetime import timedelta
from numpy.lib import format

import StringIO

from pprint import PrettyPrinter

pprinter = PrettyPrinter()

class Experiment(object):
	def __init__(self, parameters, store_prefix=''):
		self.parameters = parameters
		self.store_prefix = store_prefix

	def get_system(self):
		sys_class = self.parameters['system']
		sys = sys_class(**self.parameters.get('system_params', {}))
		return sys

	def get_solver(self):
		parameters = self.parameters
		sys = self.get_system()
		scheme_class = parameters['scheme']
		scheme = scheme_class(**parameters.get('scheme_params', {}))
		solver_class = parameters['solver']
		solver = solver_class(scheme, sys)
		return solver

	def run(self, save=True):
		parameters = self.parameters
		self.solver = self.get_solver()
		self.solver.initialize(**parameters['initialize'])
		print '-'*80
		print self.get_path()
		print parameters['name']
		pprinter.pprint(parameters)
		start_time = time.time()
		try:
			self.solver.run()
		except Exception as e:
			print e
		finally:
			end_time = time.time()
			self.duration = end_time - start_time
			print str(timedelta(seconds=self.duration))
			self.timestamp = datetime.datetime.now()
			if save:
				self.save()

	def get_path(self):
		path = os.path.join(self.store_prefix, self.parameters['family'])
		return path

	@classmethod
	def info_shelf(self, name):
		return name+'_info'

	@classmethod
	def event_shelf(self, name):
		return name+'_events'

	def save(self):
		parameters = self.parameters
		path = self.get_path()
		shelf = shelve.open(path)
		name = parameters['name']
		info = {}
		info['timestamp'] = self.timestamp
		info['duration'] = self.duration
		info['params'] = parameters
		event_string = StringIO.StringIO()
		events = self.solver.events_array
		format.write_array(event_string, events)
		shelf[self.info_shelf(name)] = info
		shelf[self.event_shelf(name)] = event_string.getvalue()
		shelf.close()

	@classmethod
	def load(self, family, name):
		shelf = shelve.open(family)
		info = shelf[self.info_shelf(name)]
		event_string = shelf[self.event_shelf(name)]
		event_file = StringIO.StringIO(event_string)
		events = format.read_array(event_file)

		params = info['params']
		experiment = self(params)
		solver = experiment.get_solver()
		solver.load_data(events)
		shelf.close()
		return solver


