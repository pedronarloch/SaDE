import math
import random
from cmath import sqrt, cos, pi, sin

import sympy
import yaml


class Problem(object):
	dimensions = 0
	problem_type = 0
	ub = None
	lb = None

	def __init__(self):
		print("Class Problem Instantied!")


class CEC(Problem):
	problem = ""
	cecScore = None

	def __init__(self):
		super().__init__()
		print("Class CEC Instantied!")
		self.readParameters()
		self.getBounds()

	def check_bounds(self, trial):
		for i in range(0, len(trial)):
			if trial[i] < self.lb[i]:
				trial[i] = random.uniform(self.lb[i], self.ub[i])
			elif trial[i] > self.ub[i]:
				trial[i] = random.uniform(self.lb[i], self.ub[i])

	def read_parameters(self):
		with open("cec_config.yaml", 'r') as stream:
			try:
				config = yaml.load(stream)
				self.problem = config['problem']
				self.dimensions = config['dimensions']
			except yaml.YAMLError as exc:
				print(exc)

	def get_bounds(self):
		self.ub = []
		self.lb = []
		if self.problem == 1 or self.problem == "Ackley":
			for i in range(0, self.dimensions):
				self.lb.append(-32.768)
				self.ub.append(32.768)
		elif self.problem == 2 or self.problem == "Griewank":
			for i in range(0, self.dimensions):
				self.lb.append(-600)
				self.ub.append(600)
		elif self.problem == 3 or self.problem == "Rastringin":
			for i in range(0, self.dimensions):
				self.lb.append(-5.12)
				self.ub.append(5.12)
		elif self.problem == 4 or self.problem == "Schwefel":
			for i in range(0, self.dimensions):
				self.lb.append(-500)
				self.ub.append(500)
		elif self.problem == 5 or self.problem == "Composite Function 4":
			for i in range(0, self.dimensions):
				self.lb.append(-5)
				self.ub.append(5)
		else:
			print("Problem not Found! Bounds not set!")
			return 0

	def evaluate(self, solution):
		if self.problem == 1 or self.problem == "Ackley":
			return self.ackley(solution)
		elif self.problem == 2 or self.problem == "Griewank":
			return self.griewank(solution)
		elif self.problem == 3 or self.problem == "Rastringin":
			return self.rastrigin(solution)
		elif self.problem == 4 or self.problem == "Schwefel":
			return self.schwefel(solution)
		else:
			print("Problem not Found!")
			return 0

	def ackley(self, solution):
		a = 20
		b = 0.2
		c = 2*pi

		d = len(solution)

		sum1 = 0
		sum2 = 0

		for i in range(0, len(solution)):
			sum1 += math.pow(solution[i], 2)
			sum2 += cos(2*pi*solution[i])

		term1 = (-a * sympy.exp(-b*sqrt(sum1/d)))
		term2 = -sympy.exp(sum2/d)

		y = term1 + term2 + a + math.exp(1)

		return y

	def griewank(self, solution):
		ii = []
		p = 1
		s = 0

		for i in range(0, len(solution)):
			ii.append(i+1)

		for i in range(0, len(solution)):
			x = cos(solution[i]/sqrt(ii[i])).real
			p = p * x
			s += math.pow(solution[i],2)/4000

		y = s - p + 1

		return y

	def rastrigin(self, solution):
		d = len(solution)
		s = 0
		y = 0

		for i in range(0, d):
			s += (math.pow(solution[i],2) - 10 * cos(2*pi*solution[i])).real

		y = 10 * d + s

		return y

	def schwefel(self, solution):
		d = len(solution)
		s = 0
		y = 0

		for i in range(0, d):
			s += (solution[i] * sin(sqrt(abs(solution[i])))).real

		y = 418.9829 * d - s

		return y