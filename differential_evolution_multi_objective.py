import copy
import math
import random

import numpy as np

import differential_evolution_ufrgs as de
from individual import MultiObjectiveIndividual


class DEMO(de.DifferentialEvolution):

    def __init__(self, problem):
        print('Differential Evolution Multi Objective Instantied')
        super().__init__(problem)

        self.pool_of_solutions = []
        self.problem.is_multi_objective = True

    def init_population(self):
        print("Initializing a Random Population")
        self.population = np.empty(self.NP, object)

        for i in range(0, self.NP):
            ind = MultiObjectiveIndividual(i, self.problem.dimensions, 2)
            ind.size = self.problem.dimensions
            ind.rand_gen(self.problem.lb, self.problem.ub)

            fitness_value = self.problem.evaluate(ind.dimensions)
            ind.fitness[0] = fitness_value[0]
            ind.fitness[1] = fitness_value[1]

            self.population[i] = ind

        self.initial_population = copy.deepcopy(self.population)

    def init_population_by_apl(self):
        print("Initializing the Population by APL")
        self.population = np.empty(self.NP, object)

        for i in range(0, self.NP):
            ind = MultiObjectiveIndividual(i, self.problem.dimensions, 2)
            ind.dimensions = np.copy(self.problem.generate_apl_individual())

            fitness_value = self.problem.evaluate(ind.dimensions)
            ind.fitness[0] = fitness_value[0]
            ind.fitness[1] = fitness_value[1]

            ind.ind_id = i

            self.population[i] = ind

        self.initial_population = copy.deepcopy(self.population)

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
        non_ranked = np.array(self.pool_of_solutions)

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
            for i in range(0, len(self.pool_of_solutions)):
                if self.pool_of_solutions[i].rank == curr_rank:
                    values.append((i, self.pool_of_solutions[i].fitness[0], self.pool_of_solutions[i].fitness[1]))

            str_arr = np.array(values, dtype=dtype)  # Structured Array based on individual information

            arr = np.sort(str_arr, order='objective_1')

            # In crowding distance, boundaries must have its distance set to Infinite.
            self.pool_of_solutions[arr[0]['index']].crowding_distance = math.inf
            self.pool_of_solutions[arr[-1]['index']].crowding_distance = math.inf

            for i in range(1, len(arr) - 1):
                distance = (arr[i + 1]['objective_1'] - arr[i - 1]['objective_1']) / \
                           (arr[-1]['objective_1'] - arr[0]['objective_1'])

                self.pool_of_solutions[arr[i]['index']].crowding_distance = distance

            arr = np.sort(str_arr, order="objective_2")
            self.pool_of_solutions[arr[0]['index']].crowding_distance = math.inf
            self.pool_of_solutions[arr[-1]['index']].crowding_distance = math.inf

            for i in range(1, len(arr) - 1):
                distance = (arr[i + 1]['objective_2'] - arr[i - 1]['objective_2']) / \
                           (arr[-1]['objective_2'] - arr[0]['objective_2'])
                self.pool_of_solutions[arr[i]['index']].crowding_distance += distance

            curr_rank += 1

    def get_highest_rank(self):
        rank = 0

        for solution in self.population:

            if solution.rank > rank:
                rank = solution.rank

        return rank

    # TODO Talvez a modificação deva ser feita aqui, atualizando o Pool of Individuals
    def generational_operator(self, gen_trial, pop_index):

        costs = np.array((gen_trial.fitness, self.population[pop_index].fitness))

        # The generated individual dominates the parent
        pareto_efficiency = self.is_pareto_efficient(costs)

        if pareto_efficiency[0] and pareto_efficiency[1]:  # Both solutions are in the optimal front (non-dominance)
            self.pool_of_solutions.append(self.population[pop_index])
            self.pool_of_solutions.append(copy.deepcopy(gen_trial))
        elif pareto_efficiency[0] and not pareto_efficiency[1]:  # Offspring dominates parent
            self.pool_of_solutions.append(gen_trial)
        elif not pareto_efficiency[0] and pareto_efficiency[1]:  # Parent dominates offspring
            self.pool_of_solutions.append(self.population[pop_index])
        else:  # Non-dominance (optimal front)
            self.pool_of_solutions.append(self.population[pop_index])
            self.pool_of_solutions.append(gen_trial)

    def truncate_offspring(self):

        dtype = [('index', int), ('rank', float), ('crowding_distance', float)]
        values = []

        for i in range(0, len(self.pool_of_solutions)):
            values.append((i, self.pool_of_solutions[i].rank, self.pool_of_solutions[i].crowding_distance))

        str_arr = np.array(values, dtype=dtype)
        str_arr = np.sort(str_arr, order="rank")

        rank_count = 0
        aux = []

        aux_offspring = []
        aux_idx = 0

        while len(aux_offspring) < self.NP:
            aux.clear()
            for i in range(aux_idx, len(str_arr)):
                if str_arr[i]['rank'] == rank_count:
                    aux.append(str_arr[i])
                    aux_idx += 1
                else:
                    break

            if len(aux) + len(aux_offspring) <= self.NP:
                aux_offspring.extend(copy.copy(aux))
            else:

                str_arr = np.array(aux, dtype=dtype)
                str_arr = np.sort(str_arr, order="crowding_distance")[::-1]
                index = 0

                while len(aux_offspring) < self.NP:
                    aux_offspring.append(str_arr[index])
                    index += 1

                break

            rank_count += 1

        for i, sol in enumerate(aux_offspring):
            self.population[i] = copy.deepcopy(self.pool_of_solutions[sol['index']])

    # TODO It is necessary to modify the operator in order to consider individuals from the population AND offspring
    def rand_1_bin(self, j, trial_individual):

        arr1 = np.array(self.new_individuals)

        pool = np.concatenate((self.population, arr1))

        while 1:
            r1 = self.selection_operator()
            if r1 != j:
                break

        while 1:
            r2 = self.selection_operator()
            if r2 != j and r2 != r1:
                break

        while 1:
            r3 = self.selection_operator()
            if r3 != r2 and r3 != r1 and r3 != j:
                break

        jRand = random.randint(0, self.problem.dimensions - 1)

        trial = trial_individual.dimensions
        r1_dimensions = pool[r1].dimensions
        r2_dimensions = pool[r2].dimensions
        r3_dimensions = pool[r3].dimensions

        for d in range(0, self.problem.dimensions):
            if random.random() <= self.CR or d == jRand:
                trial[d] = r1_dimensions[d] + (self.F * (r2_dimensions[d] - r3_dimensions[d]))

        self.problem.check_bounds(trial)

        return trial

    def selection_operator(self):
        solution_pool = len(self.new_individuals) + len(self.population)
        return random.randint(0, solution_pool)

    def optimize(self, i_pop=None):
        if i_pop is None:
            self.init_population()
        else:
            self.population = i_pop

        for i in range(0, self.MAX):

            for j in range(0, self.NP):

                trial = copy.deepcopy(self.population[j])

                self.rand_1_bin(j, trial)

                fitness_values = self.problem.evaluate(trial.dimensions)

                trial.fitness[0] = fitness_values[0]
                trial.fitness[1] = fitness_values[1]

                self.generational_operator(trial, j)

            # TODO Corrigir concatenando os indivíduos corretos
            if len(self.new_individuals) > 0:
                self.pool_of_solutions = np.concatenate((self.old_individuals, self.new_individuals))
            else:
                self.pool_of_solutions = self.population

            self.non_dominated_sorting()
            self.calculate_crowding_distance()
            self.truncate_offspring()
