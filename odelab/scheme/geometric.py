#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy as np

from . import Scheme

class SymplecticEuler(Scheme):
	def get_residual(self, t, u0, h):
		q0 = self.system.position(u0)
		p0 = self.system.momentum(u0)
		def residual(du):
			u1 = u0+du
			q1 = self.system.position(u1)
			p1 = self.system.momentum(u1)
			uu = np.hstack([q1,p0]) # only works if u = [q,p], not [q,v]...
			v = self.system.velocity(uu)
			force = self.system.force(uu)
			return np.hstack([
				q1 - q0 - h*v,
				p1 - p0 - h*force,
				])
		return residual

class StormerVerlet(Scheme):
	def get_residual(self, t,u0,h):
		q0 = self.system.position(u0)
		def residual(du):
			u1 = u0+du
			p1 = self.system.momentum(u1)
			uu = np.hstack([q0,p1])

