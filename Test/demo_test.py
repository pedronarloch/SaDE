import pytest
import numpy as np
from individual import MultiObjectiveIndividual
from differential_evolution_multi_objective import DEMO
from problem.generic_problem import Problem
from math import inf


class TestDEMO:

    @pytest.fixture
    def solution_set(self):
        population = np.empty(12, object)
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

        return population

    @pytest.fixture
    def problem(self):
        return Problem()

    @pytest.fixture
    def differential_evolution(self, problem, solution_set):

        de = DEMO(problem)
        de.NP = 12
        de.CR = 1
        de.F = 0.5
        de.MAX = 1
        de.strategy = 0
        de.population = solution_set

        return de

    def test_generational_operator_both_optimal(self, differential_evolution, solution_set):

        child = solution_set[-1]  # Fitness: [-3.5, -3.5]
        parent = solution_set[-2]  # Fitness: [-5.0, -2.0]

        differential_evolution.generational_operator(child, len(solution_set) - 2)

        assert np.array_equal(differential_evolution.pool_of_solutions[0].fitness, parent.fitness)
        assert np.array_equal(differential_evolution.pool_of_solutions[1].fitness, child.fitness)

    def test_generational_operator_child_optimal(self, differential_evolution, solution_set):

        child = solution_set[-1]  # Fitness: [-3.5, -3.5]

        differential_evolution.generational_operator(child, 0)  # Parent Fitness: [4, 2]

        assert len(differential_evolution.pool_of_solutions) == 1
        assert not np.array_equal(differential_evolution.pool_of_solutions[0].fitness, solution_set[0].fitness)
        assert np.array_equal(differential_evolution.pool_of_solutions[0].fitness, child.fitness)

    def test_generational_operator_parent_optimal(self, differential_evolution, solution_set):

        child = solution_set[0]  # Fitness: [4, 2]

        differential_evolution.generational_operator(child, -1)

        with pytest.raises(IndexError) as e_info:  # In this case, the child must be dropped
            differential_evolution.new_individuals[0]

        print(" Exception Obtained: ", e_info)

        assert np.array_equal(differential_evolution.old_individuals[0].fitness, differential_evolution.population[-1].fitness)

    def test_generational_operator_none_optimal(self, differential_evolution, solution_set):

        child = solution_set[1]  # Fitness [3,3]
        parent = solution_set[1]  # Fitness [3,3]

        differential_evolution.generational_operator(child, 1)

        assert np.array_equal(differential_evolution.old_individuals[0].fitness, parent.fitness)
        assert np.array_equal(differential_evolution.new_individuals[0].fitness, child.fitness)

    def test_non_dominated_sorting(self, differential_evolution):
        differential_evolution.non_dominated_sorting(differential_evolution.population)

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

        for i, sol in enumerate(solution_set[-5:]):
            differential_evolution.generational_operator(sol, i)

        differential_evolution.pool_of_solutions = np.concatenate((differential_evolution.old_individuals, differential_evolution.new_individuals))

        differential_evolution.non_dominated_sorting()
        differential_evolution.calculate_crowding_distance()

        differential_evolution.truncate_offspring()

        assert len(differential_evolution.population) == differential_evolution.NP

        final_population = np.empty(differential_evolution.NP, dtype=np.ndarray)

        for i, solution in enumerate(differential_evolution.population):
            assert np.array_equal(solution.fitness, solution_set[7 + i].fitness)

    def test_truncate_offspring_general(self, differential_evolution, solution_set):

        return None


    def test_rand_1_bin(self, differential_evolution, solution_set):
        return None

