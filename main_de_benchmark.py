import sys
import differential_evolution_multi_objective as demo
import problem.benchmark_problem as benchmark

problem = benchmark.BenchmarkProblem()
algorithm_de = demo.DEMO(problem)

mean_p_values = []
for i in range(0, 30):
    mean_p_values.append(algorithm_de.optimize())
    algorithm_de.dump()

print(mean_p_values)