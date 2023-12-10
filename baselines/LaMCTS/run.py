from functions.functions import *
# from functions.mujoco_functions import *
from lamcts import MCTS
import argparse
import numpy as np
import torch
import time
import random
import os
import datetime



torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)


dim_high = 22
d_e = 10
seed = 0
budget = 500
num_exp = 10
init_nums = 5
func_name = 'cassini2'
target = Cassini2Gtopx
func_val_all = torch.zeros(num_exp, budget)
func_val_all_full = torch.ones(num_exp, budget)
time_all = torch.zeros(num_exp)

folder = os.path.exists("../results")
if not folder:
    os.makedirs("../results")
path = "./results/" + func_name + "/LaMCTS_" + "D" + str(dim_high) + "_de" + str(d_e) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)


for seed in range(num_exp):
    random.seed(0)
    idx = random.sample([dim for dim in range(dim_high)], d_e)
    idx.sort()
    f = target(idx, path, init_nums, dim_high)

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = MCTS(
        lb=f.lb,  # the lower bound of each problem dimensions
        ub=f.ub,  # the upper bound of each problem dimensions
        dims=f.dims,  # the problem dimensions
        ninits=f.ninits,  # the number of random samples used in initializations
        func=f,  # function object to be optimized
        Cp=f.Cp,  # Cp for MCTS
        leaf_size=f.leaf_size,  # tree leaf size
        kernel_type=f.kernel_type,  # SVM configruation
        gamma_type=f.gamma_type  # SVM configruation
    )

    start = time.time()
    agent.search(iterations=budget)
    end = time.time()

    time_all[seed] = torch.tensor(end - start)
    #
    # fig = plt.figure(figsize=(7, 5))
    # matplotlib.rcParams.update({'font.size': 16})
    # plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
    # # Plot cumulative minimum as a red line
    # plt.plot(np.minimum.accumulate(fX), 'r', lw=3)
    # plt.xlim([0, len(fX)])
    # plt.ylim([0, 30])
    # plt.title("50D ackley function")
    #
    # plt.tight_layout()
    # plt.show()

    file_path = path + '/' + "seed" + str(seed) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__() + ".txt"

    file = open(str(file_path), 'w')
    file.write("=============================== \n")
    file.write("EX: LaMCTS \n")
    file.write("Datetime: " + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
    file.write("=============================== \n\n\n")
    file.write("=============================== \n")
    file.write("          BASIC INFOS           \n")
    file.write("=============================== \n")
    file.write("D: " + str(dim_high) + " \n")
    file.write("d_e: " + str(d_e) + " \n")
    file.write("Effective dim:" + str(idx) + " \n")
    file.write("Init points: " + str(init_nums) + " \n")
    # file.write("x*:" + str(x_best) + "\n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    file.write("=============================== \n\n\n")


f = open(path + '/result' + str(budget))
for seed in range(num_exp):
    objectives = f.readline()
    objectives = np.array([float(i) for i in objectives[1: -2].split(', ')])
    func_val_all[seed] = -torch.from_numpy(np.minimum.accumulate(objectives))
    fX = -torch.from_numpy(objectives)
    func_val_all_full[seed] = fX


best_func_val = torch.zeros(budget)
for i in range(budget):
    best_func_val[i] = func_val_all[:, i].max()
mean = torch.mean(func_val_all, dim=0)
std = torch.sqrt(torch.var(func_val_all, dim=0))
median = torch.median(func_val_all, dim=0).values
# median = np.median(func_val_all.numpy(), axis=0)
file = open(str(path + '/experiment_result=' + str(round(-float(mean[-1]), 4)) + '.txt'), 'w')
file.write(f"The best function value across all the {num_exp} experiments: \n")
file.write(str(best_func_val))
file.write(f"\n\nThe mean of the function value across all the {num_exp} experiments: \n")
file.write(str(mean))
file.write(f"\n\nThe standard deviation of the function value across all the {num_exp} experiments: \n")
file.write(str(std))
file.write(f"\n\nThe median of the function value across all the {num_exp} experiments: \n")
file.write(str(median))
file.write(f"\n\nThe mean time each experiment consumes across all the {num_exp} experiments (s): \n")
file.write(str(time_all.mean()))
torch.save(func_val_all_full, path + '/f.pt')

# #你需要把他们的函数输入转成我们的目标函数需要的，再把我们目标函数的输出转换成他们方法要求的
#
# parser = argparse.ArgumentParser(description='Process inputs')
# parser.add_argument('--func', default='ackley', help='specify the test function')
# parser.add_argument('--dims', default=50, type=int, help='specify the problem dimensions')
# parser.add_argument('--iterations', default=1000, type=int, help='specify the iterations to collect in the search')
#
# #f = ObjFunc.ackley_new
#
# args = parser.parse_args()
#
# d_e = 10
#
# random.seed(0)
# idx = random.sample([dim for dim in range(args.dims)], d_e)
# idx.sort()
#
#
# f = None
# iteration = 0
# if args.func == 'ackley':
#     assert args.dims > 0
#     f = Ackley(idx, dims=args.dims)
# elif args.func == 'griewank':
#     assert args.dims > 0
#     f = Griewank(idx, dims=args.dims)
# elif args.func == 'sphere':
#     assert args.dims > 0
#     f = Sphere(idx, dims=args.dims)
# elif args.func == 'zakharov':
#     assert args.dims > 0
#     f = Zakharov(idx, dims=args.dims)
# elif args.func == 'levy':
#     assert args.dims > 0
#     f = Levy(dims=args.dims)
# # elif args.func == 'lunar':
# #     f = Lunarlanding()
# # elif args.func == 'swimmer':
# #     f = Swimmer()
# # elif args.func == 'hopper':
# #     f = Hopper()
# else:
#     #f = ObjFunc.ackley_new
#     print('function not defined')
#     os._exit(1)
#
# assert f is not None
# assert args.iterations > 0
#
#
# # f = Ackley(dims = 10)
# # f = Levy(dims = 10)
# # f = Swimmer()
# # f = Hopper()
# # f = Lunarlanding()
#
# agent = MCTS(
#              lb = f.lb,              # the lower bound of each problem dimensions
#              ub = f.ub,              # the upper bound of each problem dimensions
#              dims = f.dims,          # the problem dimensions
#              ninits = f.ninits,      # the number of random samples used in initializations
#              func = f,               # function object to be optimized
#              Cp = f.Cp,              # Cp for MCTS
#              leaf_size = f.leaf_size, # tree leaf size
#              kernel_type = f.kernel_type, #SVM configruation
#              gamma_type = f.gamma_type    #SVM configruation
#              )
#
# agent.search(iterations = args.iterations)
#
# """
# FAQ:
#
# 1. How to retrieve every f(x) during the search?
#
# During the optimization, the function will create a folder to store the f(x) trace; and
# the name of the folder is in the format of function name + function dimensions, e.g. Ackley10.
#
# Every 100 samples, the function will write a row to a file named results + total samples, e.g. result100
# mean f(x) trace in the first 100 samples.
#
# Each last row of result file contains the f(x) trace starting from 1th sample -> the current sample.
# Results of previous rows are from previous experiments, as we always append the results from a new experiment
# to the last row.
#
# Here is an example to interpret a row of f(x) trace.
# [5, 3.2, 2.1, ..., 1.1]
# The first sampled f(x) is 5, the second sampled f(x) is 3.2, and the last sampled f(x) is 1.1
#
# 2. How to improve the performance?
# Tune Cp, leaf_size, and improve BO sampler with others.
#
# """
