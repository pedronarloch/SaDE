import os
import time
import differential_evolution_ufrgs as de
import numpy as np
from problem import molecular_docking_problem as md


def retrieve_init_pop(protein, run_id):
    ind_list = list()
    f = open("./results_canonical_de/PSP/APL/" + protein + "/" + str(run_id + 1) + "/init_pop_n", 'r')
    for line in f:
        ind = [float(vals) for vals in line.split(sep=',')]
        ind_list.append(ind)

    f.close()

    return ind_list


problem_md = md.MolecularDockingProblem()
algorithm_de = de.DifferentialEvolution(problem_md)

for i in range(0, 30):

    #pre_defined_pop = np.empty(algorithm_de.NP, individuals.Individual)
    #list_pop = retrieve_init_pop(problem_psp.protein, i)

    #for k in range(0, algorithm_de.NP):
    #    ind = individuals.Individual(k, problem_psp.dimensions)
    #    ind.ind_id = k
    #    ind.dimensions = list_pop[k]
    #    ind.fitness = problem_psp.evaluate(ind.dimensions)

    #    pre_defined_pop[k] = ind

    os.makedirs("./results/differential_evolution/" + problem_md.docking_complex + "/" + str(i + 1))
    init_time = time.time()
    file_name = "./results/differential_evolution/" + problem_md.docking_complex + "/" + str(i + 1) + "/"

    #algorithm_de.optimize(pre_defined_pop)
    algorithm_de.optimize()
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

    algorithm_de.dump()