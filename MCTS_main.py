import os
import datetime
import numpy as np
from core import ObjFunc
from core.MCTS import MCTS
from core.BayeOpt import init_points_dataset_bo
from core.RandEmbed import generate_random_matrix
from zoopt import Parameter
import torch
import time
import random



torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)


def no_growth(x, num):
    x_no = torch.ones(num) * -1e5
    # print(x)
    for i in range(x[0].shape[0]):
        x_no[i] = x[0, i]
    # x_no = x[0].clone()
    for i in range(1, x_no.shape[0]):
        x_no[i] = x_no[i-1:i+1].max()
    # print(x_no)
    return x_no


dim_high = 100
dim_low = 20
d_e = 10

Cp = 100
cp_decay_rate = 0
split_type = 'bound50%'
UCT_type = 'mean'
# max_height = torch.log(torch.tensor(dim_high))
# max_height = 500
max_height = 4

opt_method = 'bo'
budget = 500
func = ObjFunc.griewank_rand
function_name = 'ttttt'
num_exp = 20
bounds_low = torch.tensor([[-1., 1.]] * dim_low)
# bounds_high = torch.tensor([[-32.768, 32.768]] * dim_high)
bounds_high = torch.tensor([[-50., 50.]] * dim_high)
# bounds_high = torch.tensor([[-5.12, 5.12]] * dim_high)
# bounds_high = torch.tensor([[-5., 10.]] * dim_high)
# bounds_high = torch.tensor([[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
#                              1.05, 1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
#                             [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
#                              6.5, 291.0, torch.pi, torch.pi, torch.pi, torch.pi]]).T
# bounds_high = torch.tensor([[12, 161, 184, 184, 45, 119, 60, 60, 184, 119, 184, 184, 60, 184, 184, 401, 115, 84, 84, 84, 85,
#                        85, 85, 84, 60, 60, 60, 60, 48, 48, 13],
#                       [12, 15, 15, 15, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
kernel_type = 'matern'
cmaes_sigma = 0.5
if opt_method == 'racos':
    split_threshold = dim_low * 20
    pop_size = 0
    if max_height != 1:
        init_nums = Parameter(budget=split_threshold).get_train_size()
    else:
        init_nums = Parameter(budget=budget).get_train_size()
elif opt_method == 'cmaes':
    init_nums = 0
    pop_size = 5
    # split_threshold = dim_low * init_nums * 3
    split_threshold = 40 * pop_size
else:
    init_nums = 5
    split_threshold = 125
    pop_size = 0
    # split_threshold = 10

# init_nums = 5 if opt_method != 'racos' else Parameter(budget=split_threshold).get_train_size()
# positive_percentage = 0.05

# bounds_high = torch.tensor([[-50., 50.]] * dim_high)

folder = os.path.exists("../results_ablation")
if not folder:
    os.makedirs("../results_ablation")
path = "./results_ablation/" + function_name + "/MCTS_" + str(opt_method) + "_D" + str(dim_high) + "_d" + str(dim_low) + "_de" + str(d_e) + "_"\
       + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
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

    def obj_func(x):
        return func(x, idx)

    bound_vector = bounds_low[:, 1] ** 2
    # sigma = (bounds_high[:, 1] - bounds_high[:, 0]) / (2 * torch.sqrt(bound_vector.sum()))
    sigma = (bounds_high[:, 1] - bounds_high[:, 0]) / (2 * torch.log10(torch.tensor(dim_low + 1)) * torch.sqrt(bound_vector.sum()))
    # print(sigma)
    start = time.time()
    rand_mat = generate_random_matrix(dim_low, dim_high, sigma, seed)
    dataset = init_points_dataset_bo(init_nums, rand_mat, bounds_low, bounds_high, obj_func)
    solver = MCTS(Cp, cp_decay_rate, split_threshold, split_type, UCT_type, max_height, budget, init_nums, dataset,
                  rand_mat, bounds_high, obj_func, dim_high, dim_low, bounds_low, seed, opt_method, kernel_type,
                  pop_size, path, cmaes_sigma)
    # solver = MCTS(Cp, split_threshold, split_type, budget, init_nums, dataset, max_height, rand_mat,
    #               bounds_high, obj_func, dim_high, dim_low, seed, opt_method, UCT_type, cp_decay_rate, kernel_type,
    #               init_nums)
    flag, bounds = solver.search()
    end = time.time()
    # print(solver.root.optimizer, solver.root.rchild.optimizer)

    n = torch.argmax(solver.total_data['f'])
    # print(solver.total_data['x'][n], solver.total_data['f'][n])
    # print(solver.root.optimizer._parameter)
    func_val_all[seed] = no_growth(solver.total_data['f'][0:budget, :].reshape(1, -1), budget)
    func_val_all_full[seed] *= solver.total_data['f'][0:budget, :].reshape(1, -1)[0, -1]
    func_val_all_full[seed, 0:solver.total_data['f'][0:budget, :].reshape(1, -1).shape[-1]] = \
        solver.total_data['f'][0:budget, :].reshape(1, -1)
    time_all[seed] = torch.tensor(end - start)

    file_path = path + '/' + "seed" + str(seed) + "_" + \
                    datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__() + ".txt"

    file = open(str(file_path), 'w')
    file.write("=============================== \n")
    file.write("EX: MCTS \n")
    file.write("Datetime: " + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
    file.write("=============================== \n\n\n")
    file.write("=============================== \n")
    file.write("          BASIC INFOS           \n")
    file.write("=============================== \n")
    file.write("D: " + str(dim_high) + " \n")
    file.write("d: " + str(dim_low) + " \n")
    file.write("d_e: " + str(d_e) + " \n")
    file.write("Effective dim:" + str(idx) + " \n")
    file.write("Cp: " + str(Cp) + " \n")
    file.write("Cp decay rate: " + str(cp_decay_rate) + " \n")
    file.write("Optimization method: " + str(opt_method) + " \n")
    if opt_method == 'cmaes':
        file.write("CMAES pop size: " + str(pop_size) + " \n")
    file.write("Kernel method: " + str(kernel_type) + " \n")
    if flag:
        file.write("Budget: " + str(budget) + " \n")
    else:
        file.write("Budget (early stopped): " + str(budget) + "(" + str(solver.total_data['f'].shape[0]) + ") \n")
    file.write("Split threshold: " + str(split_threshold) + " \n")
    file.write("Objective Function: " + function_name + " \n")
    file.write("Split type: " + split_type + " \n")
    file.write("Max tree height: " + str(max_height) + " \n")
    file.write("UCT type: " + UCT_type + " \n")
    file.write("Init points: " + str(init_nums) + " \n")
    # file.write("Init points for new node: " + str(init_nums_new_node) + " \n")
    file.write("Random seed: " + str(seed) + " \n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    file.write("x*:" + str(solver.total_data['x'][n]) + "\n")
    file.write("optimal value:" + str(-solver.total_data['f'][n]) + "\n")
    file.write("=============================== \n\n\n")
    file.write("The bound of best node:" + "\n" + str(bounds) + "\n")
    file.write("The bound of best node (effective dimensions):" + "\n" + str(torch.index_select(bounds, 0, torch.tensor(idx))) + "\n")

    # file.write("=============================== \n")
    # file.write("        All Loop Results        \n")
    # file.write("=============================== \n")
    # file.write("\n\nThe total dataset of x is: \n")
    # file.write(str(solver.total_data['x']))
    # file.write("\n\nThe total dataset of function value is: \n")
    # file.write(str(solver.total_data['f']))
    # file.write("\n\n=============================== \n\n\n")
    torch.save(solver.total_data['x'], path + '/x' + str(seed + 1) + '.pt')

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
