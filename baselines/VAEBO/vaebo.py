# The original code is from https://github.com/lamda-bbo/MCTS-VS/tree/master/baseline

import torch
import torch.nn as nn
from torch.nn import functional as F
import botorch
import numpy as np
import pandas as pd
import random
import argparse
import datetime
import os
import time
from core import ObjFunc, BayeOpt
# from benchmark import get_problem
# from baseline.vanilia_bo import generate_initial_data, get_gpr_model, optimize_acqf
# from utils import latin_hypercube, from_unit_cube, save_results, save_args


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.latent_mapping = nn.Linear(latent_dim, 128)
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        latent_z = self.latent_mapping(z)
        out = self.decoder(latent_z)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


def train_vae(train_x, epochs=50):
    # train_x = torch.tensor(train_x, dtype=torch.float)
    print('Retraining VAE model.')
    for epoch in range(epochs):
        for idx in range(train_x.shape[0]):
            x = train_x[idx: idx + 1]
            out_x, mu, logvar = vae_model(x)
            recons_loss = F.mse_loss(out_x, x)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
            loss = recons_loss + 0.5 * kld_loss
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # print(loss)
    print('Training process completed.')
        # print(loss)


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
budget = 500
func = ObjFunc.griewank_rand
function_name = 'griewank'
num_exp = 20
init_nums = 5
kernel_type = 'matern'
lr = 1e-3
# active_dims = 6
update_interval = 20
# batch_size = 3

# bounds_high = torch.tensor([[-32.768, 32.768]] * dim_high)
# bounds_high = torch.tensor([[-50., 50.]] * dim_high)
# bounds_high = torch.tensor([[-5.12, 5.12]] * dim_high)
bounds_high = torch.tensor([[-5., 10.]] * dim_high)
# bounds_high = torch.tensor([[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
#                              1.05, 1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
#                             [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
#                              6.5, 291.0, torch.pi, torch.pi, torch.pi, torch.pi]]).T
# bounds_high = torch.tensor([[12, 15, 15, 15, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                              0, 0],
#                             [13, 161, 184, 184, 45, 119, 60, 60, 184, 119, 184, 184, 60, 184, 184, 401, 115, 84, 84, 84,
#                              85, 85, 85, 84, 60, 60, 60, 60, 48, 48, 13]]).T



folder = os.path.exists("../results")
if not folder:
    os.makedirs("../results")
path = "./results/" + function_name + "/VAE-BO_D" + str(dim_high) + "_d" + str(dim_low) + "_de" + str(d_e) + "_"\
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

    torch.manual_seed(seed)


    vae_model = VAE(dim_high, dim_low)
    opt = torch.optim.Adam(vae_model.parameters(), lr=lr)
    dataset = {'x': torch.rand(init_nums, dim_high) * (bounds_high.t()[1] - bounds_high.t()[0]) + bounds_high.t()[0]}
    dataset['f'] = obj_func(dataset['x']).reshape(init_nums, 1)

    start = time.time()
    train_data = dataset['x'].clone()
    train_data = (train_data - bounds_high.t()[0]) / (bounds_high.t()[1] - bounds_high.t()[0]) * 2 - 1
    # train_data = (train_data - bounds_high.t()[0]) / (bounds_high.t()[1] - bounds_high.t()[0])
    train_vae(train_data)
    # train_vae(dataset['x'])
    # sample_counter = init_nums

    flag = True
    for i in range(init_nums, budget):
        if i % update_interval == 0:
            # train_data = dataset['x'].clone()
            # train_data = ((train_data - bounds_high.t()[0]) / (bounds_high.t()[1] - bounds_high.t()[0])) * 2 - 1
            train_vae(train_data)

        # np_train_y = np.array(dataset['f'])
        # mu, logvar = vae_model.encode(dataset['x'])
        mu, logvar = vae_model.encode(train_data)
        z = vae_model.sample_z(mu, logvar).detach()
        # if i % update_interval == 0:
        #     print(train_data[0], vae_model.decode(z)[0])
        # z = z.numpy()
        for dim in range(z.shape[1]):
            z[:, dim] = torch.clamp(z[:, dim], torch.tensor(-1.), torch.tensor(1.))
        # z = np.clip(z, lb[0], ub[0])
        # print(z, z.shape)
        dataset['y'] = z
        # print(dataset['y'].max(dim=0), dataset['y'].min(dim=0))
        # print(dataset)
        flag, next_z = BayeOpt.next_point_bo(dataset, 0.2 * dim_low * torch.log(torch.tensor(2 * (i + 1))),
                                             torch.tensor([[-1., 1.]] * dim_low), 'matern')
        if not flag:
            break
        # model, mll = BayeOpt.fit_model_gp(dataset, 'matern')
        # next
        # gpr = get_gpr_model()
        # gpr.fit(z, np_train_y)
        # new_z, _ = optimize_acqf(args.active_dims, gpr, z, np_train_y, args.batch_size, lb[0] * args.active_dims,
        #                          ub[0] * args.active_dims)
        next_x = vae_model.decode(next_z).detach()


        # for dim in range(next_x.shape[1]):
        #     next_x[:, dim] = torch.clamp(next_x[:, dim], bounds_high[dim, 0], bounds_high[dim, 1])
        for dim in range(next_x.shape[1]):
            next_x[:, dim] = torch.clamp(next_x[:, dim], torch.tensor(-1.), torch.tensor(1.))
        train_data = torch.cat([train_data, next_x], dim=0)
        next_x = (next_x + 1) / 2 * (bounds_high.t()[1] - bounds_high.t()[0]) + bounds_high.t()[0]
        # next_x = next_x * (bounds_high.t()[1] - bounds_high.t()[0]) + bounds_high.t()[0]
        # print(next_x)
        # new_x = [np.clip(new_x[i], lb[0], ub[0]) for i in range(new_x.shape[0])]
        next_f = obj_func(next_x)
        del dataset['y']
        # print(next_x)
        dataset = BayeOpt.update_dataset_ucb(None, next_x, next_f, dataset)
        print(f'Iteration: {i}', next_f)
        # print(dataset['x'].max(dim=0), dataset['x'].min(dim=0))

        # train_x.extend(new_x)
        # train_y.extend(new_y)
        # sample_counter += len(new_x)
        # if sample_counter >= args.max_samples:
        #     break

    end = time.time()
    # print(solver.root.optimizer, solver.root.rchild.optimizer)
    # print(dataset['f'])
    n = torch.argmax(dataset['f'])
    print(f"Final best f value: {dataset['f'][n]}", )
    # print(solver.total_data['x'][n], solver.total_data['f'][n])
    # print(solver.root.optimizer._parameter)
    func_val_all[seed] = no_growth(dataset['f'][0:budget, :].reshape(1, -1), budget)
    func_val_all_full[seed] *= dataset['f'][0:budget, :].reshape(1, -1)[0, -1]
    func_val_all_full[seed, 0:dataset['f'][0:budget, :].reshape(1, -1).shape[-1]] = dataset['f'][0:budget, :].reshape(1, -1)
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
    # if opt_method == 'racos':
    #     file.write("Positive percentage: " + str(positive_percentage) + " \n")
    file.write("Kernel method: " + str(kernel_type) + " \n")
    if flag:
        file.write("Budget: " + str(budget) + " \n")
    else:
        file.write("Budget (early stopped): " + str(budget) + "(" + str(dataset['f'].shape[0]) + ") \n")
    file.write("Objective Function: " + function_name + " \n")
    file.write("Init points: " + str(init_nums) + " \n")
    # file.write("Init points for new node: " + str(init_nums_new_node) + " \n")
    file.write("Random seed: " + str(seed) + " \n")
    file.write("Total time consume: " + str(end - start) + " s \n")
    file.write("optimal value:" + str(-dataset['f'][n]) + "\n")
    file.write("=============================== \n\n")
    file.write("x*:" + str(dataset['x'][n]) + "\n")
    file.write("x* (eff dim):" + str(torch.index_select(dataset['x'][n], 0, torch.tensor(idx))))

    # file.write("=============================== \n")
    # file.write("        All Loop Results        \n")
    # file.write("=============================== \n")
    # file.write("\n\nThe total dataset of x is: \n")
    # file.write(str(solver.total_data['x']))
    # file.write("\n\nThe total dataset of function value is: \n")
    # file.write(str(solver.total_data['f']))
    # file.write("\n\n=============================== \n\n\n")
    torch.save(dataset['x'], path + '/x' + str(seed + 1) + '.pt')

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


# parser = argparse.ArgumentParser()
# parser.add_argument('--func', default='hartmann6_50', type=str)
# parser.add_argument('--max_samples', default=600, type=int)
# parser.add_argument('--init_samples', default=10, type=int)
# parser.add_argument('--batch_size', default=3, type=int)
# parser.add_argument('--update_interval', default=20, type=int)
# parser.add_argument('--active_dims', default=6, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--root_dir', default='synthetic_logs', type=str)
# parser.add_argument('--seed', default=2021, type=int)
# args = parser.parse_args()
# print(args)
#
#
# seed = 0
# random.seed(args.seed)
# np.random.seed(args.seed)
# botorch.manual_seed(args.seed)
# torch.manual_seed(args.seed)
#
# save_config = {
#     'save_interval': 50,
#     'root_dir': 'logs/' + args.root_dir,
#     'algo': 'vae_{}'.format(args.active_dims),
#     'func': args.func,
#     'seed': args.seed
# }
# func = get_problem(args.func, save_config)
# dims = func.dims
# lb = func.lb
# ub = func.ub
#
# save_args(
#     'config/' + args.root_dir,
#     'vae_{}'.format(args.active_dims),
#     args.func,
#     args.seed,
#     args
# )
#
# vae_model = VAE(func.dims, args.active_dims)
# opt = torch.optim.Adam(vae_model.parameters(), lr=args.lr)
#
# # train_x, train_y = generate_initial_data(func, args.init_samples, lb, ub)
# points = latin_hypercube(args.init_samples, dims)
# points = from_unit_cube(points, lb, ub)
# train_x, train_y = [], []
# for i in range(args.init_samples):
#     y = func(points[i])
#     train_x.append(points[i])
#     train_y.append(y)
# sample_counter = args.init_samples
# best_y = [(sample_counter, np.max(train_y))]
#
# train_vae(train_x)
#
# while True:
#     if sample_counter % args.update_interval == 0:
#         train_vae(train_x)
#
#     np_train_y = np.array(train_y)
#     mu, logvar = vae_model.encode(torch.tensor(train_x, dtype=torch.float))
#     z = vae_model.sample_z(mu, logvar)
#     z = z.detach().numpy()
#     z = np.clip(z, lb[0], ub[0])
#
#     gpr = get_gpr_model()
#     gpr.fit(z, np_train_y)
#     new_z, _ = optimize_acqf(args.active_dims, gpr, z, np_train_y, args.batch_size, lb[0] * args.active_dims,
#                              ub[0] * args.active_dims)
#     new_x = vae_model.decode(torch.tensor(new_z, dtype=torch.float))
#     new_x = new_x.detach().numpy()
#
#     new_x = [np.clip(new_x[i], lb[0], ub[0]) for i in range(new_x.shape[0])]
#     new_y = [func(x) for x in new_x]
#
#     train_x.extend(new_x)
#     train_y.extend(new_y)
#     sample_counter += len(new_x)
#     if sample_counter >= args.max_samples:
#         break
#
# print('best f(x):', func.tracker.best_value_trace[-1])