import differential_evolution_multi_objective as demo
import problem.benchmark_problem as benchmark


problem = benchmark.BenchmarkProblem()
algorithm_de = demo.DEMO(problem)

algorithm_de.optimize()


