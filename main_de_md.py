import os
import time

import numpy as np

import differential_evolution_ufrgs as de
from individual import Individual
from problem import molecular_docking_problem as md


def retrieve_init_pop(instance, run_id):

    ind_list = np.empty(100, dtype=np.ndarray)
    f = open("./initial_populations/molecular_docking/" + instance + "/" + str(run_id + 1) + "/init_pop", 'rb')

    for i, line in enumerate(f):
        line = line.translate(None, delete=b'[]\n')
        ind_list[i] = np.fromstring(line.decode('utf-8'), sep=',')

    f.close()

    initial_ligand = np.empty(1, dtype=np.ndarray)
    f = open("./initial_populations/molecular_docking/"+instance+"/"+str(run_id+1)+"/init_ligand", 'rb')

    for line in f:
        line = line.translate(None, delete=b'[]\n')
        initial_ligand = np.fromstring(line.decode('utf-8'), sep=',')

    return ind_list, initial_ligand


problem_md = md.MolecularDockingProblem()
algorithm_de = de.DifferentialEvolution(problem_md)


for i in range(0, 1):

    pre_defined_pop = np.empty(algorithm_de.NP, Individual)
    list_pop, init_ligand = retrieve_init_pop(problem_md.docking_complex, i)
    problem_md.randomize_ligand(init_ligand)

    for k in range(0, algorithm_de.NP):
        ind = Individual(k, problem_md.dimensionality)
        ind.ind_id = k
        ind.dimensions = list_pop[k]
        ind.fitness = problem_md.evaluate(ind.dimensions)

        pre_defined_pop[k] = ind

    os.makedirs("./results/differential_evolution/" + problem_md.docking_complex + "/" + str(algorithm_de.strategy)
                + "/" + str(i + 1))
    init_time = time.time()
    file_name = "./results/differential_evolution/" + problem_md.docking_complex + "/" + str(algorithm_de.strategy) \
                + "/" + str(i + 1) + "/"

    algorithm_de.optimize(pre_defined_pop)

    # algorithm_de.optimize()
    algorithm_de.get_results_report(file_name + "convergence")
    end_time = time.time()

    t_file = open(file_name + "run_time", "w")
    t_file.write(str(end_time - init_time) + " seconds")
    t_file.close()

    init_ligand_pos = open(file_name+"init_ligand", "w")
    init_ligand_pos.write(str(problem_md.random_initial_dimensions))
    init_ligand_pos.close()

    init_pop = open(file_name+"init_pop", "w")
    for j in range(0, len(algorithm_de.initial_population)):
        init_pop.write(np.array2string(algorithm_de.initial_population[j].dimensions, max_line_width=999999, precision=4,
                                       separator=","))
        init_pop.write("\n")
    init_pop.close()

    final_pop = open(file_name + "final_pop", "w")
    for j in range(0, len(algorithm_de.population)):
        final_pop.write(
            np.array2string(algorithm_de.population[j].dimensions, max_line_width=999999, precision=4,
                            separator=","))
        final_pop.write("\n")
    final_pop.close()

    best_individual = algorithm_de.population[algorithm_de.get_best_individual()]

    best_ind = open(file_name+"best_conformation", "w")
    best_ind.write(np.array2string(best_individual.dimensions, max_line_width=999999, precision=4,
                                   separator=","))
    best_ind.close()
    problem_md.evaluate(best_individual.dimensions)
    problem_md.energy_function.dump_ligand_pdb(file_name+"best_individual.pdb")

    algorithm_de.dump()
