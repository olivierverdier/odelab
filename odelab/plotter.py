# -*- coding: UTF-8 -*-
from __future__ import division

import matplotlib.pyplot as PL
import numpy as np

class Plotter(object):

	max_plot_res = 500 # max plot resolution

	def __init__(self, solver):
		self.solver = solver
		self.system = self.solver.system

	def sample(self, plot_res=None):
		plot_res = plot_res or self.max_plot_res
		# some sampling
		size = len(self.solver)
		stride = np.ceil(size/self.max_plot_res)
		self.events = self.solver.events_array[:,::stride]

	def components(self, components, time_component):
		# components
		if components is None: # plot all the components by default
			components = range(self.solver.initial().size - 1)
		# makes the code work when plotting a single component (either a string or an index):
		if isinstance(components, basestring) or not np.iterable(components):
			components = [components]
		if time_component is not None:
			components.insert(0,time_component)
		return components

	def generate_plot_data(self, components=None, plot_exact=True, error=False, save=None, time_component=None):
		self.sample()
		events = self.events
		ats = events[-1]

		components = self.components(components, time_component)

		if plot_exact or error:
			sys_exact = getattr(self.system, 'exact', None)
			if sys_exact:
				exact = sys_exact(ats, self.solver.initial())
		compute_exact = (plot_exact or error) and sys_exact

		time_label = 'time'

		for component_i, component in enumerate(components):
			if isinstance(component, str):
				label = component
				function = getattr(self.system, component)
				data = function(events)
				if compute_exact:
					exact_comp = function(np.vstack([exact, ats]))
			else:
				label = self.system.label(component)
				data = events[component]
				if compute_exact:
					exact_comp = exact[component]

			if error and sys_exact:
				data = np.log10(np.abs(data - exact_comp))
			if time_component is not None and not component_i:
				# at the first step, if time_component is not time, then replace the time vector by the desired component
				ats = data
				time_label = label
				continue
			else:
				pts = {}
				pts['x'] = ats
				pts['time_label'] = time_label
				pts['data'] = data
				pts['label'] = label
				if compute_exact and not error:
					pts['exact'] = exact_comp
				yield pts

	def plot(self, components=None, plot_exact=True, error=False, save=None, time_component=None, **plot_args):
		"""
		Plot some components of the solution.

:param list components: either a given component of the solution, or a list of components to plot, or a list of strings corresponding to methods of the system.
:param boolean plot_exact: whether to plot the exact solution (if available)
:param string save: whether to save the plot in a file
:param boolean error: to plot (the log10 of) the error instead of the value itself
:param int time_component: component to use as x variable (if None, then time is used)
		"""
		# set up plot arguments
		defaults = {'ls':'-', 'marker':','}
		defaults.update(plot_args)

		axis = PL.gca()
		if save: # if this is meant to be saved, clear the previous plot first
			axis.cla()
		for pts in self.generate_plot_data(components, plot_exact, error, save, time_component):
			data = pts['data']
			ats = pts['x']
			label = pts['label']
			axis.plot(ats, data, ',-', label=label,  **defaults)
			exact = pts.get('exact', None)
			if exact is not None:
				current_color = axis.lines[-1].get_color() # figure out current colour
				axis.plot(ats, exact, ls='-', lw=2, label='{}*'.format(label), color=current_color)

		axis.set_xlabel(pts['time_label'])
		axis.legend()
		if save:
			PL.savefig(save, format='pdf', **plot_args)
		else:
			PL.plot() # plot only in interactive mode
		return axis

