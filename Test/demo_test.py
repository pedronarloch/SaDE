from math import inf

import numpy as np
import pytest

from differential_evolution_multi_objective import DEMO
from individual import MultiObjectiveIndividual
from problem.generic_problem import Problem


class TestDEMO:

    @pytest.fixture
    def solution_set(self):
        population = np.empty(12, object)
        for i in range(0, 12):
            population[i] = MultiObjectiveIndividual(i, 2, 2)
            population[i].dimensions = [1, 1, 1]

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

        return population

    @pytest.fixture
    def problem(self):
        problem = Problem()
        problem.ub = [1, 1, 1]
        problem.lb = [1, 1, 1]
        return problem

    @pytest.fixture
    def differential_evolution(self, problem, solution_set):
        problem.dimensionality = 3
        de = DEMO(problem)
        de.NP = 12
        de.CR = 1
        de.F = 0.5
        de.MAX = 1
        de.strategy = 0
        de.population = solution_set

        return de

    def test_generational_operator_non_dominance(self, differential_evolution, solution_set):
        child = solution_set[-1]  # Fitness: [-3.5, -3.5]
        parent = solution_set[-2]  # Fitness: [-5.0, -2.0]

        differential_evolution.pool_of_solutions.append(parent)
        differential_evolution.generational_operator(child, len(solution_set) - 2)

        assert np.array_equal(differential_evolution.pool_of_solutions[0].fitness, parent.fitness)
        assert np.array_equal(differential_evolution.pool_of_solutions[1].fitness, child.fitness)

    def test_generational_operator_child_dominance(self, differential_evolution, solution_set):

        child = solution_set[-1]  # Fitness: [-3.5, -3.5]
        parent = solution_set[0]
        differential_evolution.pool_of_solutions.append(parent)
        differential_evolution.generational_operator(child, 0)  # Parent Fitness: [4, 2]

        assert len(differential_evolution.pool_of_solutions) == 1
        assert not np.array_equal(differential_evolution.pool_of_solutions[0].fitness, solution_set[0].fitness)
        assert np.array_equal(differential_evolution.pool_of_solutions[0].fitness, child.fitness)

    def test_generational_operator_parent_dominance(self, differential_evolution, solution_set):

        child = solution_set[0]  # Fitness: [4, 2]
        parent = solution_set[-1]
        differential_evolution.pool_of_solutions.append(parent)
        differential_evolution.generational_operator(child, -1)

        assert np.array_equal(differential_evolution.pool_of_solutions[0].fitness, solution_set[-1].fitness)

    def test_generational_operator_non_dominance_equals(self, differential_evolution, solution_set):

        child = solution_set[1]  # Fitness [3,3]
        parent = solution_set[1]  # Fitness [3,3]
        differential_evolution.pool_of_solutions.append(parent)
        differential_evolution.generational_operator(child, 1)

        assert np.array_equal(differential_evolution.pool_of_solutions[0].fitness, parent.fitness)
        assert np.array_equal(differential_evolution.pool_of_solutions[1].fitness, child.fitness)

    def test_non_dominated_sorting(self, differential_evolution):
        differential_evolution.pool_of_solutions = differential_evolution.population
        differential_evolution.non_dominated_sorting()

        correct_ranks = [2, 2, 2, 3, 3, 4, 1, 0, 0, 0, 0, 0]
        de_ranks = []

        for ind in differential_evolution.population:
            de_ranks.append(ind.rank)

        assert correct_ranks == de_ranks

    def test_calculate_crowding_distance(self, differential_evolution, solution_set):
        differential_evolution.NP = 5
        differential_evolution.population = solution_set[7:]
        differential_evolution.pool_of_solutions = differential_evolution.population

        differential_evolution.non_dominated_sorting()
        differential_evolution.calculate_crowding_distance()

        correct_distances = ["{:.1f}".format(inf), "{:.1f}".format(1.0), "{:.1f}".format(1.0), "{:.1f}".format(inf), "{:.1f}".format(0.66666)]
        de_distances = []

        for ind in differential_evolution.pool_of_solutions:
            de_distances.append("{:.1f}".format(ind.crowding_distance))

        assert correct_distances == de_distances

    def test_truncate_offspring_optimal_equal_pop_size(self, differential_evolution, solution_set):

        differential_evolution.NP = 5
        differential_evolution.population = solution_set[:5]
        differential_evolution.pool_of_solutions = list(differential_evolution.population)

        for i, sol in enumerate(solution_set[-5:]):
            differential_evolution.generational_operator(sol, i)

        differential_evolution.non_dominated_sorting()
        differential_evolution.calculate_crowding_distance()

        differential_evolution.truncate_offspring()

        assert len(differential_evolution.population) == differential_evolution.NP

        for i, solution in enumerate(differential_evolution.population):
            assert np.array_equal(solution.fitness, solution_set[7 + i].fitness)

    def test_truncate_offspring_add_all(self, differential_evolution, solution_set):

        differential_evolution.pool_of_solutions = differential_evolution.population
        differential_evolution.non_dominated_sorting()
        differential_evolution.calculate_crowding_distance()

        differential_evolution.truncate_offspring()

        dtype = [('index', int), ('rank', int), ('fitness', np.ndarray)]
        values = []
        values_set = []

        assert len(differential_evolution.population) == differential_evolution.NP

        for i, solution in enumerate(differential_evolution.population):
            values.append((i, solution.rank, solution.fitness))

        for i, sol in enumerate(solution_set):
            values_set.append((i, sol.rank, sol.fitness))

        str_arr = np.array(values, dtype=dtype)
        str_arr = np.sort(str_arr, order="rank")

        str_arr_aux = np.array(values, dtype=dtype)
        str_arr_aux = np.sort(str_arr_aux, order="rank")

        assert np.array_equal(str_arr, str_arr_aux)

    def test_truncate_offspring_different_ranks(self, differential_evolution, solution_set):
        differential_evolution.NP = 9
        differential_evolution.population = solution_set[:9]

        differential_evolution.pool_of_solutions = solution_set

        differential_evolution.non_dominated_sorting()
        differential_evolution.calculate_crowding_distance()
        differential_evolution.truncate_offspring()

        dtype = [('index', int), ('rank', int), ('fitness', np.ndarray)]
        values = []
        values_set = []

        assert len(differential_evolution.population) == differential_evolution.NP

        for i, solution in enumerate(differential_evolution.population):
            values.append((i, solution.rank, solution.fitness))

        for i, sol in enumerate(solution_set[:3]):
            values_set.append((i, sol.rank, sol.fitness))

        for i, sol in enumerate(solution_set[6:]):
            values_set.append((i, sol.rank, sol.fitness))

        str_arr = np.array(values, dtype=dtype)
        str_arr = np.sort(str_arr, order="rank")

        str_arr_aux = np.array(values, dtype=dtype)
        str_arr_aux = np.sort(str_arr_aux, order="rank")

        assert np.array_equal(str_arr, str_arr_aux)