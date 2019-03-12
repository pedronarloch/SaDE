import sys

import numpy as np

import differential_evolution_multi_objective
from individual import MultiObjectiveIndividual


def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
    return is_efficient


if __name__ == '__main__':

    population = np.empty(12, object)
    demo = differential_evolution_multi_objective.DEMO()
    demo.NP = 8
    demo.dump()

    for i in range(0, 12):
        population[i] = MultiObjectiveIndividual(i, 2, 2)

    population[0].fitness = np.array([4, 2])
    population[1].fitness = np.array([3, 3])
    population[2].fitness = np.array([2, 4])
    population[3].fitness = np.array([5, 3])
    population[4].fitness = np.array([3, 5])
    population[5].fitness = np.array([7, 5])
    population[6].fitness = np.array([-2, -2])
    population[7].fitness = np.array([-2, -5])
    population[8].fitness = np.array([-3, -4])
    population[9].fitness = np.array([-4, -3])
    population[10].fitness = np.array([-5, -2])
    population[11].fitness = np.array([-3.5, -3.5])

    demo.population = population
    demo.non_dominated_sorting()
    demo.calculate_crowding_distance()
    demo.temp_offspring = population

    for i in range(0, len(demo.population)):
        print(demo.population[i].rank, demo.population[i].crowding_distance)

    demo.truncate_offspring()

    sys.exit()



