import copy
import math
import random
import pygmo

import numpy as np

import differential_evolution_ufrgs as de
from individual import MultiObjectiveIndividual


class DEMO(de.DifferentialEvolution):

    def __init__(self, problem):
        print('Differential Evolution Multi Objective Instantied')
        super().__init__(problem)

        self.pool_of_solutions = []
        self.new_individuals = []
        self.problem.is_multi_objective = True

    def dump(self):
        super().dump()
        self.pool_of_solutions = []
        self.new_individuals = []

    def init_population(self):
        print("Initializing a Random Population")
        self.population = np.empty(self.NP, object)

        for i in range(0, self.NP):
            ind = MultiObjectiveIndividual(i, self.problem.dimensionality, self.problem.get_num_objectives())
            ind.size = self.problem.dimensionality
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
            ind = MultiObjectiveIndividual(i, self.problem.dimensionality)
            ind.dimensions = np.copy(self.problem.generate_apl_individual())

            fitness_value = self.problem.evaluate(ind.dimensions)
            ind.fitness[0] = fitness_value[0]
            ind.fitness[1] = fitness_value[1]

            ind.ind_id = i

            self.population[i] = ind

        self.initial_population = copy.deepcopy(self.population)

    def non_dominated_sorting(self):
        points = []

        for i, candidate in enumerate(self.pool_of_solutions):
            points.append([float(candidate.fitness[0]), float(candidate.fitness[1])])

        ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(points)

        for i, rank in enumerate(ndr):
            self.pool_of_solutions[i].rank = rank

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

            # In crowding distance, boundaries must have their distance set to Infinite.
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

        for solution in self.pool_of_solutions:

            if solution.rank > rank:
                rank = solution.rank

        return rank

    def generational_operator(self, gen_trial, pop_index):

        # If the mutated vector dominates the parent, it takes the parent's position in the offspring
        if pygmo.pareto_dominance(gen_trial.fitness, self.population[pop_index].fitness):
            self.pool_of_solutions[pop_index] = copy.deepcopy(gen_trial)
        elif pygmo.pareto_dominance(self.population[pop_index].fitness, gen_trial.fitness):  # The parent dominates the mutant vector
            return
        else:  # If there is not dominance between solutions, both of them goes to the offspring vector
            self.pool_of_solutions.append(copy.deepcopy(gen_trial))

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

        self.population = np.empty(self.NP, object)
        for i, sol in enumerate(aux_offspring):
            self.population[i] = copy.deepcopy(self.pool_of_solutions[sol['index']])
        self.pool_of_solutions = []

    def rand_1_bin(self, j, trial_individual):

        while 1:
            r1, r1_dimensions = self.selection_operator()
            if r1 != j:
                break

        while 1:
            r2, r2_dimensions = self.selection_operator()
            if r2 != j and r2 != r1:
                break

        while 1:
            r3, r3_dimensions = self.selection_operator()
            if r3 != r2 and r3 != r1 and r3 != j:
                break

        jRand = random.randint(0, self.problem.dimensionality - 1)

        trial = trial_individual.dimensions

        for d in range(0, self.problem.dimensionality):
            if random.random() <= self.CR or d == jRand:
                trial[d] = r1_dimensions[d] + (self.F * (r2_dimensions[d] - r3_dimensions[d]))

        self.problem.check_bounds(trial)

        return trial

    def selection_operator(self):
        index_to_return = random.randint(0, (len(self.pool_of_solutions) - 1))

        return index_to_return, self.pool_of_solutions[index_to_return].dimensions

    def optimize(self, i_pop=None):
        if i_pop is None:
            self.init_population()
        else:
            self.population = i_pop

        for i in range(0, self.MAX):

            if i % 5 == 0:
                print('Generation ', i)

            self.pool_of_solutions = list(self.population)

            for j in range(0, self.NP):

                trial = copy.deepcopy(self.population[j])

                self.rand_1_bin(j, trial)

                fitness_values = self.problem.evaluate(trial.dimensions)

                trial.fitness[0] = fitness_values[0]
                trial.fitness[1] = fitness_values[1]

                self.generational_operator(trial, j)

            self.non_dominated_sorting()
            self.calculate_crowding_distance()
            self.truncate_offspring()

        mean_p_distance = 0
        for i, solution in enumerate(self.population):
            print(self.problem.return_p_distance(solution.dimensions))
            mean_p_distance += self.problem.return_p_distance(solution.dimensions)

        return mean_p_distance/self.NP
