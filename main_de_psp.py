import os
import sys
import time

import numpy as np

import differential_evolution_multi_objective as demo
import individual as individuals
import problem.protein_structure_prediction_problem as psp


def retrieve_init_pop(protein, run_id):
    ind_list = list()
    f = open("./results_canonical_de/PSP/APL/" + protein + "/" + str(run_id + 1) + "/init_pop_n", 'r')
    for line in f:
        ind = [float(vals) for vals in line.split(sep=',')]
        ind_list.append(ind)

    f.close()

    return ind_list


problem_psp = psp.ProteinStructurePredictionProblem()
algorithm_de = demo.DEMO(problem_psp)

sys.exit()


for i in range(0, 30):

    pre_defined_pop = np.empty(algorithm_de.NP, individuals.Individual)
    list_pop = retrieve_init_pop(problem_psp.protein, i)

    for k in range(0, algorithm_de.NP):
        ind = individuals.Individual(k, problem_psp.dimensions)
        ind.ind_id = k
        ind.dimensions = list_pop[k]
        ind.fitness = problem_psp.evaluate(ind.dimensions)

        pre_defined_pop[k] = ind

    os.makedirs("./results/differential_evolution/" + problem_psp.protein + "/SaDE/" + str(i + 1))
    init_time = time.time()
    file_name = "./results/differential_evolution/" + problem_psp.protein + "/SaDE/" + str(i + 1) + "/"

    algorithm_de.optimize(pre_defined_pop)
    algorithm_de.get_results_report(file_name + "convergence")
    end_time = time.time()

    t_file = open(file_name + "run_time", "w")
    t_file.write(str(end_time - init_time) + " seconds")
    t_file.close()

    problem_psp.generate_pdb(algorithm_de.population[algorithm_de.get_best_individual()].dimensions, file_name + "lowest_final_energy.pdb")

    algorithm_de.dump()
