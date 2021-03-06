import random
from decimal import getcontext

import numpy as np


class Individual(object):

	def __init__(self, ind_id, dimensionality):
		self.size = dimensionality
		self.dimensions = np.empty(dimensionality)
		self.fitness = 0.0
		self.indId = ind_id

	def rand_gen(self, lb, ub):
		for i in range(0, self.size):
			self.dimensions[i] = random.uniform(lb[i], ub[i])


class ClusteredIndividual(Individual):
	cluster_id = 0

	def __init__(self, ind_id, dimensionality):
		super().__init__(ind_id, dimensionality)
		self.cluster_id = 0


class MultiObjectiveIndividual(Individual):
	fitness = None

	def __init__(self, ind_id, dimensionality, number_of_objectives):
		super().__init__(ind_id, dimensionality)
		self.fitness = np.zeros(number_of_objectives)
		self.crowding_distance = 999.99
		self.is_dominated = False
		self.rank = 999


class Particle(Individual):
	getcontext().prec = 5
	p_best = None
	v = None

	p_best_fitness = 0.0

	def __init__(self, ind_id, dimensionality):
		super().__init__(ind_id, dimensionality)
		self.p_best = np.zeros(self.size)
		self.p_best_fitness = 0.0
		self.v = np.zeros(self.size)


class ClusteredParticle(Particle):
	cluster_id = 0.0

	def __init__(self, ind_id, dimensionality):
		super().__init__(ind_id, dimensionality)
		self.cluster_id = 0.0
