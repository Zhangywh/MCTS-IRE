# The original code is from https://github.com/lamda-bbo/MCTS-VS

import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import datetime
import time
import os
# from benchmark import get_problem
from MCTSVS.MCTS import MCTS
from core import ObjFunc
import functions
# from utils import save_args


dim_high = 10
dim_low = 5
d_e = 10
budget = 100
# func = ObjFunc.ackley_rand
func_name = 'test'
target = functions.Ackley
num_exp = 2
init_nums = 5

feature_batch_size = 2
sample_batch_size = 3
# min_num_variables = 3
min_num_variables = dim_low
select_right_threshold = 5
turbo_max_evals = 50
k = 20
Cp = 0.1
ipt_solver = 'bo'
uipt_solver = 'bestk'
root_dir = 'synthetic_logs'


folder = os.path.exists("../results")
if not folder:
    os.makedirs("../results")
path = "./results/" + func_name + "/MCTSVS_" + "D" + str(dim_high) + "_de" + str(d_e) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

func_val_all = torch.zeros(num_exp, budget)
func_val_all_full = torch.ones(num_exp, budget)
time_all = torch.zeros(num_exp)


for seed in range(num_exp):
    random.seed(0)
    idx = random.sample([dim for dim in range(dim_high)], d_e)
    idx.sort()
    f = target(idx, path, init_nums, dim_high)
    # f = target(dim_high)

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = MCTS(
        func=f,
        dims=f.dims,
        lb=f.lb,
        ub=f.ub,
        feature_batch_size=feature_batch_size,
        sample_batch_size=sample_batch_size,
        Cp=Cp,
        min_num_variables=min_num_variables,
        select_right_threshold=select_right_threshold,
        k=k,
        split_type='mean',
        ipt_solver=ipt_solver,
        uipt_solver=uipt_solver,
        turbo_max_evals=turbo_max_evals,
    )

    start = time.time()
    agent.search(max_samples=budget, verbose=False)
    end = time.time()
    print(agent.value_trace)
    print('best f(x):', agent.value_trace[-1][1])

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
    # func_val_all[seed] = -torch.from_numpy(np.maximum.accumulate(objectives))
    func_val_all[seed] = torch.from_numpy(objectives)
    # fX = -torch.from_numpy(objectives)
    # func_val_all_full[seed] = fX


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
torch.save(func_val_all, path + '/f.pt')
# parser = argparse.ArgumentParser()
# parser.add_argument('--func', default='hartmann6_300', type=str,
#                     choices=['hartmann6_300', 'hartmann6_500', 'levy10_100', 'levy10_300', 'nasbench', 'nasbench201', 'nasbench1shot1', 'nasbenchtrans', 'nasbenchasr', 'Hopper', 'Walker2d'])
# parser.add_argument('--max_samples', default=600, type=int)
# parser.add_argument('--feature_batch_size', default=2, type=int)
# parser.add_argument('--sample_batch_size', default=3, type=int)
# parser.add_argument('--min_num_variables', default=3, type=int)
# parser.add_argument('--select_right_threshold', default=5, type=int)
# parser.add_argument('--turbo_max_evals', default=50, type=int)
# parser.add_argument('--k', default=20, type=int)
# parser.add_argument('--Cp', default=0.1, type=float)
# parser.add_argument('--ipt_solver', default='bo', type=str)
# parser.add_argument('--uipt_solver', default='bestk', type=str)
# parser.add_argument('--root_dir', default='synthetic_logs', type=str)
# parser.add_argument('--dir_name', default=None, type=str)
# parser.add_argument('--postfix', default=None, type=str)
# parser.add_argument('--seed', default=2021, type=int)
# args = parser.parse_args()

# print(args)
# for seed in range(num_exp):
#     random.seed(seed)
#     np.random.seed(seed)
#     botorch.manual_seed(seed)
#     torch.manual_seed(seed)
#
#     algo_name = 'mcts_vs_{}'.format(ipt_solver)
#     # if args.postfix is not None:
#     #     algo_name += ('_' + args.postfix)
#     # save_config = {
#     #     'save_interval': 50,
#     #     'root_dir': 'logs/' + args.root_dir,
#     #     'algo': algo_name,
#     #     'func': args.func if args.dir_name is None else args.dir_name,
#     #     'seed': args.seed
#     # }
#     # f = get_problem(args.func, save_config, args.seed)
#
#     # save_args(
#     #     'config/' + args.root_dir,
#     #     algo_name,
#     #     args.func if args.dir_name is None else args.dir_name,
#     #     args.seed,
#     #     args
#     # )
#
#     agent = MCTS(
#         func=func,
#         dims=func.dims,
#         lb=func.lb,
#         ub=func.ub,
#         feature_batch_size=feature_batch_size,
#         sample_batch_size=sample_batch_size,
#         Cp=Cp,
#         min_num_variables=min_num_variables,
#         select_right_threshold=select_right_threshold,
#         k=k,
#         split_type='mean',
#         ipt_solver=ipt_solver,
#         uipt_solver=uipt_solver,
#         turbo_max_evals=turbo_max_evals,
#     )
#
#     agent.search(max_samples=args.max_samples, verbose=False)
#
#     print('best f(x):', agent.value_trace[-1][1])
