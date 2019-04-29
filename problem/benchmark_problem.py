from pygmo import *

udp = dtlz(prob_id = 1)
pop = population(prob = udp, size = 105)
algo = algorithm(moead(gen = 100))

for i in range(10):
    pop = algo.evolve(pop)
    print(udp.p_distance(pop))

hv = hypervolume(pop)
print(hv.compute(ref_point=[4.524422, 0.000000, 0.000000]))
