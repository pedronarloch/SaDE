import differential_evolution_ufrgs as de
import numpy as np


class DEMO(de.DifferentialEvolution):

    def __init__(self):
        super().__init__(None)
        print('Differential Evolution Multi Objective Instantied')

    def is_pareto_efficient(costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
        return is_efficient

    def calculate_crowding_distance(self):
        return None

    def optimize(self, i_pop=None):

        return None