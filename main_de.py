import os
import sys
import time

import numpy as np
import problems.protein_structure_prediction_problem as psp

import individual as individuals
import self_adaptive_differential_evolution as sade


def retrieve_init_pop(protein, run_id):
    ind_list = list()
    f = open("./results_canonical_de/PSP/APL/" + protein + "/" + str(run_id + 1) + "/init_pop_n", 'r')
    for line in f:
        ind = [float(vals) for vals in line.split(sep=',')]
        ind_list.append(ind)

    f.close()

    return ind_list


problem_psp = psp.ProteinStructurePredictionProblem()
algorithm_de = sade.SADE(problem_psp)
algorithm_de.optimize()

dtype = [('run', int), ('final_energy', float), ('final_rmsd', float), ('final_gdt', float), ('final_tm', float)]
result_arr = np.ndarray(30, dtype=dtype)

for i in range(0, 30):

    result_arr[i]['run'] = i

    pre_defined_pop = np.empty(algorithm_de.NP, individuals.Individual)
    list_pop = retrieve_init_pop(problem_psp.protein, i)

    for k in range(0, algorithm_de.NP):
        ind = individuals.Individual(k, problem_psp.dimensions)
        ind.ind_id = k
        ind.dimensions = list_pop[k]
        ind.fitness = problem_psp.evaluate(ind.dimensions)

        pre_defined_pop[k] = ind

    os.makedirs(
        "./results/differential_evolution/" + problem_psp.protein + "/" + str(algorithm_de.strategy) + "/" + str(i + 1))
    init_time = time.time()
    file_name = "./results/differential_evolution/" + problem_psp.protein + "/" + str(
        algorithm_de.strategy) + "/" + str(i + 1) + "/"
    algorithm_de.seed = i
    # algorithm_de.optimize(pre_defined_pop)

    algorithm_de.optimize()
    end_time = time.time()

    algorithm_de.get_results_report(file_name + "convergence")

    t_file = open(file_name + "run_time", "w")
    t_file.write(str(end_time - init_time) + " seconds")
    t_file.close()

    init_pop = open(file_name + "init_pop", "w")
    for j in range(0, len(algorithm_de.initial_population)):
        init_pop.write(np.array2string(algorithm_de.initial_population[j].dimensions, separator=","))
        init_pop.write("\n")
    init_pop.close()

    rmsd_file = open(file_name + "ind_rmsd", "w")
    rmsd_file.write("ind\tenergy\trmsd\n")
    for j in range(0, len(algorithm_de.population)):
        pose = problem_psp.generate_pdb(algorithm_de.population[j].dimensions, file_name + "final-" + str(j) + ".pdb")
        rmsd = problem_psp.compare_rmsd_rcsb(pose)
        rmsd_file.write(
            str(j) + "\t" + str(problem_psp.evaluate(algorithm_de.population[j].dimensions)) + "\t" + str(rmsd) + "\n")
    rmsd_file.close()

    best_final_individual = algorithm_de.population[algorithm_de.get_best_individual()]

    problem_psp.generate_pdb(best_final_individual.dimensions, file_name +
                             "best_predicted_individual.pdb")

    result_arr[i]['final_energy'] = best_final_individual.fitness

    best_final_pose = problem_psp.create_pose_centroid(best_final_individual.dimensions)

    result_arr[i]['final_rmsd'] = problem_psp.compare_rmsd_rcsb(best_final_pose)

    algorithm_de.dump()
    sys.exit()
