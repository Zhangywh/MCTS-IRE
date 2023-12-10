from turbo import Turbo1
import numpy as np
import torch
import math
import random
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from functions import functions
import time

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


dim = 100
d_e = 10
seed = 0
budget = 500
num_exp = 20
init_nums = 5
func_name = 'sphere'
func_val_all = torch.zeros(num_exp, budget)
func_val_all_full = torch.ones(num_exp, budget)
time_all = torch.zeros(num_exp)

folder = os.path.exists("../results")
if not folder:
    os.makedirs("../results")
path = "./results/" + func_name + "/TuRBO_" + "D" + str(dim) + "_de" + str(d_e) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)


for seed in range(num_exp):
    random.seed(0)
    idx = random.sample([dim for dim in range(dim)], d_e)
    idx.sort()
    f = functions.Sphere(idx, dim)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start = time.time()
    turbo1 = Turbo1(
        f=f,  # Handle to objective function
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        n_init=init_nums,  # Number of initial bounds from an Latin hypercube design
        max_evals=budget - init_nums,  # Maximum number of evaluations
        batch_size=10,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    turbo1.optimize()
    end = time.time()

    X = turbo1.X  # Evaluated points
    fX = turbo1.fX  # Observed values



    ind_best = np.argmin(fX)   #求最小值
    f_best, x_best = fX[ind_best], X[ind_best, :]

    print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" %
          (f_best, np.around(x_best, 3)))
    X = torch.from_numpy(X)[0:budget, :]
    fX = -torch.from_numpy(fX)[0:budget, :]
    # print(X.shape, fX.shape)
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
    func_val_all[seed] = no_growth(fX.reshape(1, -1), budget)
    func_val_all_full[seed] *= fX.reshape(1, -1)[0, -1]
    func_val_all_full[seed, 0:fX[0:budget, :].reshape(1, -1).shape[-1]] = fX[0:budget, :].reshape(1, -1)
    time_all[seed] = torch.tensor(end - start)

    file_path = path + '/' + "seed" + str(seed) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__() + ".txt"

    file = open(str(file_path), 'w')
    file.write("=============================== \n")
    file.write("EX: TuRBO \n")
    file.write("Datetime: " + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
    file.write("=============================== \n\n\n")
    file.write("=============================== \n")
    file.write("          BASIC INFOS           \n")
    file.write("=============================== \n")
    file.write("D: " + str(dim) + " \n")
    file.write("d_e: " + str(d_e) + " \n")
    file.write("Effective dim:" + str(idx) + " \n")
    file.write("Init points: " + str(init_nums) + " \n")
    file.write("x*:" + str(x_best) + "\n")
    file.write("optimal value:" + str(f_best) + "\n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    file.write("=============================== \n\n\n")
    torch.save(X, path + '/x' + str(seed + 1) + '.pt')

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

# torch.save(torch.from_numpy(X), path + '/x.pt')
# torch.save(torch.from_numpy(fX), path + '/f.pt')

