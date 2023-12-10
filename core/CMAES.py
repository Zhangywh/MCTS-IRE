from cmaes import CMA
# import cma
import numpy as np
import torch
from core import ObjFunc
import os
from core.BayeOpt import init_points_dataset_bo, random_embedding
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class cmaes():
    def __init__(self, d, pop_size, bounds, sigma=1.3, seed=0):
        self.model = CMA(mean=np.zeros(d), sigma=sigma, bounds=bounds, seed=seed, population_size=pop_size)
        self.best_x = None
        self.best_f = 1e5

    def gen_next_point(self):
        points = [self.model.ask() for _ in range(self.model.population_size)]
        return True, points, [torch.unsqueeze(torch.from_numpy(point), dim=0).float() for point in points]

    def update(self, points, values):
        solutions = [(points[i], values[i]) for i in range(self.model.population_size)]
        for i, value in enumerate(values):
            if value < self.best_f:
                self.best_f = value
                self.best_x = points[i]
        self.model.tell(solutions)
# class cmaes():
#     def __init__(self, pop_size, bounds, budget, init_data, sigma=1., seed=0):
#         np.random.seed(seed)
#         cma_opts = {
#             'seed': seed,
#             'popsize': pop_size,
#             'maxiter': budget,
#             'verbose': -1,
#             'bounds': [-1 * float(bounds[0][-1]), 1 * float(bounds[0][-1])]
#         }
#         n = torch.argmax(init_data['f'])
#         self.model = cma.CMAEvolutionStrategy(init_data['y'][n].tolist(), sigma, inopts=cma_opts)
#         self.model.ask()
#         self.model.tell(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
#         # self.model = CMA(mean=np.zeros(d), sigma=sigma, bounds=bounds, seed=seed, population_size=pop_size)
#         # self.best_x = None
#         # self.best_f = 1e5
#
#     def gen_next_point(self):
#         points = self.model.ask()
#         # points = [self.model.ask() for _ in range(self.model.population_size)]
#         return True, points, [torch.unsqueeze(torch.from_numpy(point), dim=0).float() for point in points]
#
#     def update(self, points, values):
#         self.model.tell(points, values)

        # solutions = [(points[i], values[i]) for i in range(self.model.population_size)]
        # for i, value in enumerate(values):
        #     if value < self.best_f:
        #         self.best_f = value
        #         self.best_x = points[i]
        # self.model.tell(solutions)


if __name__ == "__main__":
    from core import ObjFunc
    import random
    import datetime

    def no_growth(x, num):
        x_no = torch.ones(num) * -1e5
        # print(x)
        for i in range(x[0].shape[0]):
            x_no[i] = x[0, i]
        # x_no = x[0].clone()
        for i in range(1, x_no.shape[0]):
            x_no[i] = x_no[i - 1:i + 1].max()
        # print(x_no)
        return x_no
    # from core import RandEmbed
    #
    # sigma = [7.3271] * 100
    #
    # rand_mat = RandEmbed.generate_random_matrix(20, 100, sigma, 0)
    #
    #
    # def _ackley(x):
    #     # x = solution.get_x()
    #     x = torch.from_numpy(x)
    #     x = torch.unsqueeze(x, dim=0).float()
    #     # print(x)
    #     # print(x, rand_mat)
    #     x = RandEmbed.random_embedding(x, rand_mat, torch.tensor([[-32.768, 32.768]] * 100))
    #     res = -ObjFunc.ackley(x)
    #     # bias = 0.2
    #     # value = -20 * np.exp(-0.2 * np.sqrt(sum([(i - bias) * (i - bias) for i in x]) / len(x))) - \
    #     #         np.exp(sum([np.cos(2.0*np.pi*(i-bias)) for i in x]) / len(x)) + 20.0 + np.e
    #     return float(res)
    #
    #
    # cma_opts = {
    #     'seed': 0,
    #     'popsize': 20,
    #     'maxiter': 10000,
    #     'verbose': -1,
    #     'bounds': [-1, 1]
    # }
    # init_data = init_points_dataset_bo(20, rand_mat, torch.tensor([[-1., 1.]] * 20), torch.tensor([[-32.768, 32.768]] * 100), ObjFunc.ackley)
    # n = torch.argmax(init_data['f'])
    # ori = cma.CMAEvolutionStrategy(init_data['y'][n].tolist(), 1.0, inopts=cma_opts)
    # ori.ask()
    # # print(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
    # ori.tell(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
    # for i in range(500):
    #     if i % 50 == 0:
    #         print(i, ori.result.fbest)
    #     points = ori.ask()
    #     fit = [_ackley(point) for point in points]
    #     ori.tell(points, fit)
    # print(ori.result.fbest)
    #
    # ori = cma.CMAEvolutionStrategy([0] * 20, 1.0, inopts=cma_opts)
    # ori.ask()
    # # print(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
    # ori.tell(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
    # for i in range(500):
    #     if i % 50 == 0:
    #         print(i, ori.result.fbest)
    #     points = ori.ask()
    #     fit = [_ackley(point) for point in points]
    #     ori.tell(points, fit)
    # print(ori.result.fbest)
    #
    # ori = cma.CMAEvolutionStrategy([0] * 20, 1.0, inopts=cma_opts)
    # # ori.ask()
    # # # print(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
    # # ori.tell(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
    # for i in range(500):
    #     if i % 50 == 0:
    #         print(i, ori.result.fbest)
    #     points = ori.ask()
    #     fit = [_ackley(point) for point in points]
    #     ori.tell(points, fit)
    # print(ori.result.fbest)
    #
    #
    #
    # opt = cmaes(20, 20, torch.tensor([[-1., 1.]] * 20), 10000, init_data, 1.0, 0)
    # for i in range(500):
    #     if i % 50 == 0:
    #         print(i)
    #     flag, points, data = opt.gen_next_point()
    #     next_x = [random_embedding(next_y, rand_mat, torch.tensor([[-32.768, 32.768]] * 100)) for next_y in data]
    #     next_x = torch.cat(next_x, dim=0)
    #     next_y = torch.cat(data, dim=0)
    #     next_f = ObjFunc.ackley(next_x).reshape(-1, 1)
    #     opt.update(points, [-float(f.clone()) for f in next_f])
    # print(opt.model.result.fbest)
    dim_high = 22
    d_e = 10
    num_exp = 20
    budget = 500
    pop_size = 5
    sigma = 0.5
    function_name = 'cassini2'
    obj_func = ObjFunc.cassini2_gtopx
    random.seed(0)
    # bounds_high = np.array([[-32.768, 32.768]] * dim_high)
    # bounds_high = np.array([[-50., 50.]] * dim_high)
    # bounds_high = np.array([[-5.12, 5.12]] * dim_high)
    # bounds_high = np.array([[-5., 10.]] * dim_high)
    bounds_high = np.array([[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
                             1.05, 1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
                            [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0,
                             6.5, 291.0, torch.pi, torch.pi, torch.pi, torch.pi]]).T
    folder = os.path.exists("../results")
    if not folder:
        os.makedirs("../results")
    path = "./results/" + function_name + "/CMAES_" + "D" + str(dim_high) + "_de" + str(d_e) + "_" \
           + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    idx = random.sample([dim for dim in range(dim_high)], d_e)
    idx.sort()
    func_val_all = torch.zeros(num_exp, budget)
    func_val_all_full = torch.ones(num_exp, budget)
    # total_data = torch.zeros(num_exp, 500)
    for i in range(num_exp):
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        optimizer = CMA(mean=bounds_high.mean(axis=1), sigma=sigma, bounds=bounds_high, population_size=pop_size)
        min_value = 1e5
        data = []
        # init = [np.random.uniform(-1, 1, (dim)) for _ in range(20)]
        # optimizer.tell([(x, _ackley(x)) for x in init])
        for generation in range(budget // pop_size):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = -obj_func(torch.from_numpy(x).reshape(1, -1), idx)
                if value < min_value:
                    min_value = value
                solutions.append((x, value))
                print(f"#{generation}: {value}")
            optimizer.tell(solutions)
            data += [solution[1] for solution in solutions]
        print('final', min_value)
        func_val_all[i] = no_growth(-torch.tensor([data]), budget)
        func_val_all_full[i] = -torch.tensor(data)
    # solver = cmaes(dim, 20, np.array([[-1, 1]] * dim))
    # for generation in range(500):
    #     _, points, _ = solver.gen_next_point()
    #     values = [_ackley(point) for point in points]
    #     solver.update(points, values)
    # print('ziji final', solver.best_f)
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
    # file.write(f"\n\nThe mean time each experiment consumes across all the {num_exp} experiments (s): \n")
    # file.write(str(time_all.mean()))
    torch.save(func_val_all_full, path + '/f.pt')

