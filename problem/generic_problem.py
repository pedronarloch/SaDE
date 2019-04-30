import random


class Problem(object):

    def __init__(self):
        print("Class Problem Instantied!")
        self.dimensionality = 0
        self.problem_type = 0
        self.ub = None
        self.lb = None

    def get_bounds(self):
        return None

    def evaluate(self, trial):
        return None

    def check_bounds(self, trial):
        for i, value in trial:
            if trial[i] < self.lb[i]:
                trial[i] = random.uniform(self.lb[i], self.ub[i])
            elif trial[i] > self.ub[i]:
                trial[i] = random.uniform(self.lb[i], self.ub[i])
