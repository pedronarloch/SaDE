import importlib

import pygmo
import yaml

from problem.generic_problem import Problem


class BenchmarkProblem(Problem):

    def __init__(self):
        super().__init__()
        self.class_type = ''
        self.dimensionality = 0
        self.read_parameters()

        self.problem_class = getattr(importlib.import_module("pygmo"), self.class_type)
        self.problem_object = None

        self.set_problem_object()
        self.get_bounds()

    def read_parameters(self):
        with open("benchmark_config.yaml", 'r') as stream:
            try:
                config = yaml.load(stream)
                self.class_type = config['class_type']
                self.problem_id = config['problem_id']
                self.dimensionality = config['dimensionality']
            except yaml.YAMLError as exc:
                print(exc)

    def set_problem_object(self):
        if self.class_type == 'zdt':
            self.problem_object = pygmo.problem(self.problem_class(self.problem_id, self.dimensionality))
        else:
            self.problem_object = pygmo.problem(self.problem_class(dim=self.dimensionality))

    def evaluate(self, trial):
        if self.problem_object.get_nobj() == 1:
            return self.problem_object.fitness(trial)[0]
        else:
            return self.problem_object.fitness(trial)

    def get_bounds(self):
        boundary = self.problem_object.get_bounds()
        self.lb = boundary[0]
        self.ub = boundary[1]

    def get_num_objectives(self):
        return self.problem_object.get_nobj()

    def get_p_distance(self, trial):
        return self.problem_object.p_distance(trial)
