import copy
import random
import numpy as np
import yaml
import sys
import differential_evolution_ufrgs as de


class SADE(de.DifferentialEvolution):
    LP = 0  # Learning stage generations
    CRm = 0.5  # Crossover memory
    CRs = []
    mutation_quantity = 4
    ns = np.zeros(mutation_quantity)  # sucessfull rating
    nf = np.zeros(mutation_quantity)  # fails rating
    probs = np.zeros(mutation_quantity)  # probabilities


    def __init__(self, problem):
        print("Self Adaptive Differential Evolution Instancied with Problem: " + str(type(problem)))
        self.problem = problem
        self.read_parameters()
        for i in range(0, self.mutation_quantity):
            self.probs[i] = 1 / self.mutation_quantity

    def read_parameters(self):
        with open("de_config.yaml", 'r') as stream:
            try:
                config = yaml.load(stream)
                self.NP = config['np']
                self.F = config['f']
                self.CR = config['cr']
                self.MAX = config['maxIteractions']
                self.LP = config['lp']

            except yaml.YAMLError as exc:
                print(exc)

    def learning_process(self):
        acum_prob = 0
        self.dump()

        self.init_population_by_apl()

        for i in range(0, self.LP):

            self.best_ind[i] = self.population[self.get_best_individual()]
            self.diversity[i] = self.update_diversity()

            if i % 25 == 0 and i > 0:
                self.CRm = sum(self.CRs) / len(self.CRs)
                self.CRs.clear()

            if i % 5 == 0:
                self.CR = -1
                while self.CR < 0:
                    self.CR = random.gauss(self.CRm, 0.1)

                print("Learning Generation: ", i, " Energy: ", self.best_ind[i].fitness, " Diversity: ",
                      self.diversity[i], " Pop Len: ", len(self.population), " Strategy: ", self.strategy,
                      " CRm: ", self.CRm)

            for j in range(0, self.NP):

                rand_strategy = random.uniform(0, 1)
                acum_prob = 0

                for sp in range(0, self.mutation_quantity):

                    acum_prob += self.probs[sp]

                    if rand_strategy < acum_prob:
                        # print("Estratégia Selecionada: " + str(sp))
                        self.strategy = sp
                        break

                self.F = -1
                while self.F < 0:
                    self.F = random.gauss(0.5, 0.3)
                # Evolve process, it returns True if the new individual is better than the older one. False otherwise
                result = self.evolve(j)

                if result:
                    self.ns[self.strategy] += 1
                    self.CRs.append(self.CR)
                else:
                    self.nf[self.strategy] += 1

            self.update_probabilities()
            self.population = np.empty(self.NP, object)
            self.population = np.copy(self.offspring)
            self.offspring = np.empty(self.NP, object)

    def update_probabilities(self):
        total = 0
        local_percent = []
        for i in range(0, self.mutation_quantity):
            if self.ns[i] + self.nf[i] != 0:
                local_percent.append(self.ns[i] / (self.ns[i] + self.nf[i]))
                total += local_percent[i]
            else:
                print("O metodo " + str(i) + " obteve apenas falhas: " + str(self.nf[i]))
                local_percent.append(0)

        for i in range(0, self.mutation_quantity):
            self.probs[i] = local_percent[i] / total

    def optimize(self, i_pop=None):

        self.learning_process()  # Learning phase

        self.m_nmdf = 0
        self.CRm = 0.5
        self.CRs.clear()
        self.dump()

        self.init_population_by_apl()

        for i in range(0, self.MAX):
            self.best_ind[i] = self.population[self.get_best_individual()]
            self.diversity[i] = self.update_diversity()

            if i % 25 == 0 and i > 0:
                self.CRm = sum(self.CRs) / len(self.CRs)
                self.CRs.clear()

            if i % 5 == 0:
                self.CR = -1
                while self.CR < 0:
                    self.CR = random.gauss(self.CRm, 0.1)

                print("Generation: ", i, " Energy: ", self.best_ind[i].fitness, " Diversity: ", self.diversity[i],
                      " Pop Len: ", len(self.population), " Strategy: ", self.strategy, " CRm: ", self.CRm, " F: ",
                      self.F)

            for j in range(0, self.NP):

                rand_strategy = random.uniform(0, 1)
                acum_prob = 0

                for sp in range(0, self.mutation_quantity):

                    acum_prob += self.probs[sp]

                    if rand_strategy < acum_prob:
                        self.strategy = sp
                        break

                self.F = -1
                while self.F < 0:
                    self.F = random.gauss(0.5, 0.3)

                result = self.evolve(j)

                if result:
                    self.ns[self.strategy] += 1
                    self.CRs.append(self.CR)
                else:
                    self.nf[self.strategy] += 1

            self.update_probabilities()

            self.population = np.empty(self.NP, object)
            self.population = np.copy(self.offspring)
            self.offspring = np.empty(self.NP, object)

    def evolve(self, j):

        trial = copy.deepcopy(self.population[j])

        if self.strategy == 0:
            self.rand_1_bin(j=j, trial_individual=trial)
        elif self.strategy == 1:
            self.curr_to_rand(j=j, trial_individual=trial)
        elif self.strategy == 2:
            self.best_1_bin(j=j, trial_individual=trial)
        elif self.strategy == 3:
            self.curr_to_best(j=j, trial_individual=trial)
        else:
            self.rand_1_bin(j, trial)

        trial.fitness = self.problem.evaluate(trial.dimensions)

        if trial.fitness <= self.population[j].fitness:
            self.offspring[j] = trial
            return True
        else:
            self.offspring[j] = copy.copy(self.population[j])
            return False
