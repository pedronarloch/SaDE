import copy
import math
import numpy as np

import differential_evolution_ufrgs as de


class DEMO(de.DifferentialEvolution):

    def __init__(self):
        # super().__init__(None)
        print('Differential Evolution Multi Objective Instantied')

    def is_pareto_efficient(self, costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))

        return is_efficient

    def non_dominated_sorting(self):
        rank = 0
        non_ranked = np.array(self.population)

        while non_ranked.size != 0:

            costs = np.empty([len(non_ranked), 2], dtype=float)

            for i in range(0, len(non_ranked)):
                costs[i] = non_ranked[i].fitness

            # True for Non-Dominated Solutions, False for Dominated/Equal Solutions
            pareto_efficiency = self.is_pareto_efficient(costs)

            ranked_solutions = non_ranked[pareto_efficiency]

            for p in ranked_solutions:
                p.rank = rank

            non_ranked = non_ranked[np.invert(pareto_efficiency)]
            rank += 1

        return self.population

    def calculate_crowding_distance(self):
        '''
        The comparison must be done accordingly to the individuals rank in pareto-front.
        The distance must be calculated based on each objective function and summed in the end.
        '''
        curr_rank = 0
        last_rank = self.get_highest_rank()
        values = []

        dtype = [('index', int), ('objective_1', float), ('objective_2', float)]

        while curr_rank <= last_rank:

            values.clear()
            for i in range(0, len(self.population)):
                if self.population[i].rank == curr_rank:
                    values.append((i, self.population[i].fitness[0], self.population[i].fitness[1]))

            str_arr = np.array(values, dtype=dtype)  # Structured Array based on individual information

            arr = np.sort(str_arr, order='objective_1')
            # In crowding distance, boundaries must have its distance set to Infinite.
            self.population[arr[0]['index']].crowding_distance = math.inf
            self.population[arr[-1]['index']].crowding_distance = math.inf

            for i in range(1, len(arr) - 1):
                distance = (arr[i + 1]['objective_1'] - arr[i - 1]['objective_1']) / \
                           (arr[-1]['objective_1'] - arr[0]['objective_1'])

                self.population[arr[i]['index']].crowding_distance = distance

            arr = np.sort(str_arr, order="objective_2")
            self.population[arr[0]['index']].crowding_distance = math.inf
            self.population[arr[-1]['index']].crowding_distance = math.inf

            for i in range(1, len(arr) - 1):
                distance = (arr[i + 1]['objective_2'] - arr[i - 1]['objective_2']) / \
                           (arr[-1]['objective_2'] - arr[0]['objective_2'])
                self.population[arr[i]['index']].crowding_distance += distance

            curr_rank += 1

    def get_highest_rank(self):
        rank = 0

        for solution in self.population:

            if solution.rank > rank:
                rank = solution.rank

        return rank

    def generational_operator(self, gen_trial, pop_index):

        return None

    def truncate_offspring(self):

        dtype = [('index', int), ('rank', float), ('crowding_distance', float)]
        values = []
        for i in range(0, len(self.temp_offspring)):
            values.append((i, self.population[i].rank, self.population[i].crowding_distance))

        str_arr = np.array(values, dtype=dtype)
        arr = np.sort(str_arr, order="rank")

        for i in range(len(arr) - 1):

            if arr[i]['rank'] < arr[i+1]['rank']:
                self.offspring[i] = self.population[arr[i]['index']]

            elif arr[i]['rank'] == arr[i+1]['rank'] \
                    and arr[i]['crowding_distance'] > arr[i+1]['crowding_distance']:
                self.offspring[i] = self.population[arr[i]['index']]



        return None

    # ToDo It is needed to truncate the population by the dominance order and crowding distance
    def truncate_population(self):
        return None

    def optimize(self, i_pop=None):
        if i_pop is None:
            self.init_population()
        else:
            self.population = i_pop

        for i in range(0, self.MAX):
            self.best_ind[i] = self.population[self.get_best_individual()]
            self.diversity[i] = self.update_diversity()

            if i % 5 == 0:
                print("Generation: ", i, "Fitness: ", self.best_ind[i].fitness, " Diversity: ", self.diversity[i])

            for j in range(0, self.NP):
                trial = copy.deepcopy(self.population[j])

                self.rand_1_bin(j, trial)

                trial.fitness = self.problem.evaluate(trial.dimensions)  # Not created yet

                self.generational_operator(trial, self.population[j])

            self.population = np.empty(self.NP, object)
            self.population = np.copy(self.offspring)
            self.offspring = np.empty(self.NP, object)

        return None
