import neat.population as pop
import neat.experiments.recsys.config as c
from neat.visualize import draw_net

from multiprocessing import Pool

num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000


# max_num_generations = 100000


class SharedData:
    def __init__(self):
        self.inputs_tr = None
        self.targets_tr = None
        self.input_tst = None
        self.targets_tst = None
        self.max_fitness = None


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch import autograd

    import pandas
    import numpy
    import scipy.sparse
    import scipy.sparse.linalg
    from sklearn.metrics import mean_absolute_error

    print("======[Starting neat config for Recsys]====")
    data_dir = "data/ml-100k/"
    data_shape = (943, 1682)

    data_limit = 100

    df_tr = pandas.read_csv(data_dir + "ua.base", sep="\t", header=None)
    values_tr = df_tr.values
    values_tr[:, 0:2] -= 1  # zero based ids

    df_tst = pandas.read_csv(data_dir + "ua.test", sep="\t", header=None)
    values_tst = df_tst.values
    values_tst[:, 0:2] -= 1  # zero based ids

    shared_data = SharedData()

    #########
    # Version 1: Output is of the same dimenstioanlity as input, but the missing values should be reconstructed

    #########
    # Version 2:
    shared_data.inputs_tr = list(map(lambda s: autograd.Variable(torch.Tensor([s.tolist()])), values_tr[:, 0:2]))
    if data_limit >= 0:
        shared_data.inputs_tr = shared_data.inputs_tr[:data_limit]
    shared_data.targets_tr = list(map(lambda s: autograd.Variable(torch.Tensor([[s.tolist()]])), values_tr[:, 2]))
    if data_limit >= 0:
        shared_data.targets_tr = shared_data.targets_tr[:data_limit]

    shared_data.inputs_tst = list(map(lambda s: autograd.Variable(torch.Tensor([s.tolist()])), values_tst[:, 0:2]))
    if data_limit >= 0:
        shared_data.inputs_tst = shared_data.inputs_tst[:data_limit]
    shared_data.targets_tst = list(map(lambda s: autograd.Variable(torch.Tensor([[s.tolist()]])), values_tst[:, 2]))
    if data_limit >= 0:
        shared_data.targets_tst = shared_data.targets_tst[:data_limit]

    shared_data.max_fitness = 0.0  # A MSE like fitness
    for target in shared_data.targets_tst:
        shared_data.max_fitness += float(torch.sum(target.mul(target)))

    shared_data.max_fitness /= len(shared_data.targets_tst)
    print("Maximum fitness: %f" % shared_data.max_fitness)

# RecsysConfig.FITNESS_THRESHOLD = RecsysConfig.max_fitness * .9

if __name__ == "__main__":
    NUM_WORKERS = 10
    pool = Pool(NUM_WORKERS)
    for i in range(1):
        neat = pop.Population(c.RecsysConfig)
        solution, generation = neat.run(pool, shared_data)

        if solution is not None:
            avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
            min_num_generations = min(generation, min_num_generations)

            num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
            avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (
                        num_of_solutions + 1)
            min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
            max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)
            if num_hidden_nodes == 1:
                found_minimal_solution += 1

            num_of_solutions += 1
            draw_net(solution, view=True, filename='./images/solution-' + str(num_of_solutions), show_disabled=True)

    print('Total Number of Solutions: ', num_of_solutions)
    print('Average Number of Hidden Nodes in a Solution', avg_num_hidden_nodes)
    print('Solution found on average in:', avg_num_generations, 'generations')
    print('Minimum number of hidden nodes:', min_hidden_nodes)
    print('Maximum number of hidden nodes:', max_hidden_nodes)
    print('Minimum number of generations:', min_num_generations)
    print('Found minimal solution:', found_minimal_solution, 'times')
