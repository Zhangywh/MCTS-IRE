import os
import datetime
import time

from core import ObjFunc
import core
from core.MCTS import MCTS
from core.BayeOpt import init_points_dataset_bo, next_point_bo, update_dataset_ucb
from core.RandEmbed import generate_random_matrix, random_embedding
import torch
import gpytorch
import numpy as np
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


# seed = 42
# for seed in range(1, 10):
# torch.manual_seed(seed)
# bounds = torch.tensor([[-32.768, 32.768]] * 2)
dim_high = 100
dim_low = 20
d_e = 10
target_func = ObjFunc.ackley_rand
function_name = 'ackley'
sigma = [1.0] * dim_high
init_points = 5
iter_nums = 500
num_exp = 20
kernel_type = 'matern'
# bounds_low = torch.tensor([[-1., 1.]] * dim_low)
bounds_low = torch.tensor([[-torch.sqrt(torch.tensor(dim_low)), torch.sqrt(torch.tensor(dim_low))]] * dim_low)
# bounds_high = torch.tensor([[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05,
#                              1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
#                             [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5,
#                              291.0, torch.pi, torch.pi, torch.pi, torch.pi]]).T
bounds_high = torch.tensor([[-32.768, 32.768]] * dim_high)
# bounds_high = torch.tensor([[-50., 50.]] * dim_high)
# bounds_high = torch.tensor([[-5.12, 5.12]] * dim_high)
# bounds_high = torch.tensor([[-5., 10.]] * dim_high)

folder = os.path.exists("../results")
if not folder:
    os.makedirs("../results")
path = "./results/" + function_name + "/REMBO_" + "D" + str(dim_high) + "_d" + str(dim_low) + "_de" + str(d_e) + "_"\
       + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

func_val_all = torch.zeros(num_exp, iter_nums)
func_val_all_full = torch.ones(num_exp, iter_nums)
time_all = torch.zeros(num_exp)

for seed in range(num_exp):
    random.seed(0)
    idx = random.sample([dim for dim in range(dim_high)], d_e)
    idx.sort()

    # rescale to [-1, 1]
    def obj_func(x):
        # new_range = bounds_high.t()[1] - bounds_high.t()[0]
        # new_range = new_range.reshape(1, -1)
        # mid_point = bounds_high.t().sum(dim=0) / 2 - bounds_high.t()[0]
        # x = torch.mul((x + 1) / 2, torch.cat([new_range for _ in range(x.shape[0])], dim=0)) - mid_point
        # return target_func(x, idx)
        new_range = bounds_high.t()[1] - bounds_high.t()[0]
        new_range = new_range.reshape(1, -1)
        x = torch.mul((x + 1) / 2, torch.cat([new_range for _ in range(x.shape[0])], dim=0)) + bounds_high.t()[0]
        return target_func(x, idx)

    torch.manual_seed(seed)
    rand_mat = generate_random_matrix(dim_low, dim_high, sigma, seed)
    dataset = init_points_dataset_bo(init_points, rand_mat, bounds_low, torch.tensor([[-1., 1.]] * dim_high), obj_func)
    start = time.time()
    # model, _ = fit_model_gp(dataset)
    for i in range(iter_nums - init_points):
        beta = 0.2 * dim_low * torch.log(torch.tensor(2 * (i + 1)))
        flag, next_y = next_point_bo(dataset, beta, bounds_low, kernel_type)
        if flag:
            next_x = random_embedding(next_y, rand_mat, torch.tensor([[-1., 1.]] * dim_high))
            # print(next_x.max(), next_x.min())
            next_f = obj_func(next_x)
            dataset = update_dataset_ucb(next_y, next_x, next_f, dataset)
            print(f'Iteration: {i}', next_f)
        else:
            break
    end = time.time()

    n = torch.argmax(dataset['f'])
    print(f"Final best f value: {dataset['f'][n]}")
    func_val_all[seed] = no_growth(dataset['f'].reshape(1, -1), iter_nums)
    func_val_all_full[seed] *= dataset['f'].reshape(1, -1)[0, -1]
    func_val_all_full[seed, 0:dataset['f'].shape[0]] = dataset['f'].reshape(1, -1)
    time_all[seed] = torch.tensor(end - start)

    file_path = path + '/' + "seed" + str(seed) + "_" + \
                    datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__() + ".txt"

    file = open(str(file_path), 'w')
    file.write("=============================== \n")
    file.write("EX: HesBO \n")
    file.write("Datetime: " + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
    file.write("=============================== \n\n\n")
    file.write("=============================== \n")
    file.write("          BASIC INFOS           \n")
    file.write("=============================== \n")
    file.write("D: " + str(dim_high) + " \n")
    file.write("d: " + str(dim_low) + " \n")
    file.write("d_e: " + str(d_e) + " \n")
    file.write("Objective Function: " + function_name + " \n")
    file.write("iteration times (truly used): " + str(iter_nums) + " (" + str(i + 1) + ")" + " \n")
    file.write("init points number: " + str(init_points) + " \n")
    file.write("random seed: " + str(seed) + " \n")
    # file.write("y*:" + str(dataset['y'][n]) + "\n")
    # file.write("x*:" + str(dataset['x'][n]) + "\n")
    file.write("optimal value:" + str(-dataset['f'][n]) + "\n")
    file.write("random matrix: \n" + str(rand_mat) + " \n")
    # file.write("low bounds: \n" + str(bounds_low) + " \n")
    # file.write("high bounds: \n" + str(bounds_high) + " \n")
    file.write("=============================== \n\n\n")

    # file.write("=============================== \n")
    # file.write("        All Loop Results        \n")
    # file.write("=============================== \n")
    # file.write("\n\nThe total dataset of y is: \n")
    # file.write(str(dataset['y']))
    # file.write("\n\nThe total dataset of x is: \n")
    # file.write(str(dataset['x']))
    # file.write("\n\nThe total dataset of function value is: \n")
    # file.write(str(dataset['f']))
    # file.write("\n\n=============================== \n\n\n")

best_func_val = torch.zeros(iter_nums)
for i in range(iter_nums):
    best_func_val[i] = func_val_all[:, i].max()
mean = torch.mean(func_val_all, dim=0)
std = torch.sqrt(torch.var(func_val_all, dim=0))
median = torch.median(func_val_all, dim=0).values
# median = np.median(func_val_all.numpy(), axis=0)
file = open(str(path + '/experiment_result.txt'), 'w')
file.write(f"The best function value across all the {num_exp} experiments: \n")
file.write(str(best_func_val))
file.write(f"\n\nThe mean of the function value across all the {num_exp} experiments: \n")
file.write(str(mean))
file.write(f"\n\nThe standard deviation of the function value across all the {num_exp} experiments: \n")
file.write(str(std))
file.write(f"\n\nThe median of the function value across all the {num_exp} experiments: \n")
file.write(str(median))
torch.save(func_val_all_full, path + '/f.pt')
