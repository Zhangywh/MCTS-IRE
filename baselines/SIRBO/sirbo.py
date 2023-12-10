# The original code is from https://github.com/cjfcsjt/SILBO

import numpy as np
import functions
import GPy
from scipy.linalg import eigh
import cma
import acquisition
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def Run_Main(low_dim=2, high_dim=20, initial_n=20, total_itr=100, func_type='Branin', s=None, active_var=None,
             ARD=False, variance=1., length_scale=None, noise_var=0):
    if active_var is None:
        active_var = np.arange(high_dim)
    slice_number = low_dim + 1

    # Specifying the type of objective function
    if func_type == 'Branin':
        test_func = functions.Branin(active_var, noise_var=noise_var)
    elif func_type == 'Hartmann6':
        test_func = functions.Hartmann6(active_var, noise_var=noise_var)
    elif func_type == 'rosenbrock':
        test_func = functions.Rosenbrock(high_dim, active_var)
    elif func_type == 'ackley':
        test_func = functions.Ackley(high_dim, active_var)
    elif func_type == 'sphere':
        test_func = functions.Sphere(high_dim, active_var)
    elif func_type == 'griewank':
        test_func = functions.Griewank(high_dim, active_var)
    elif func_type == 'cassini':
        test_func = functions.Cassini2Gtopx(high_dim, active_var)
    else:
        TypeError('The input for func_type variable is invalid, which is', func_type)
        return

    best_results = np.zeros([1, total_itr + initial_n])

    # generate embedding matrix via samples
    f_s_true = test_func.evaluate_true(s)
    # print(f_s_true)
    B = SIR(low_dim, s, f_s_true, slice_number)

    embedding_sample = np.matmul(s, B)
    # print(embedding_sample)
    for i in range(initial_n):
        best_results[0, i] = np.max(f_s_true[0:i + 1])
    for i in range(initial_n):
        best_results[0, i] = np.max(f_s_true[0:i + 1])

    # Standardize function values
    mean = f_s_true.mean()
    std = f_s_true.std()
    std = 1.0 if std < 1e-6 else std
    f = (f_s_true - mean) / std

    # Generating GP model
    k = GPy.kern.Matern52(input_dim=low_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    # m = GPy.models.GPRegression(embedding_sample, f_s_true, kernel=k)
    m = GPy.models.GPRegression(embedding_sample, f, kernel=k)
    m.likelihood.variance = 1e-6
    ac = acquisition.ACfunction(B, m, initial_size=initial_n, low_dimension=low_dim)

    for i in range(total_itr):
        # Updating GP model
        # m.set_XY(embedding_sample, f_s_true)
        m.set_XY(embedding_sample, f)
        m.optimize()

        es = cma.CMAEvolutionStrategy(high_dim * [0], 0.5, {'bounds': [-1, 1], 'verbose': -1})
        iter = 0
        while not es.stop() and iter != 50:
        # while not es.stop():
            iter += 1
            X = es.ask()
            # print([ac.acfunctionUCB(x)[0] for x in X])
            es.tell(X, [ac.acfunctionUCB(x)[0] for x in X])
        # maxx = es.result[0]
        maxx = es.result.xbest


        s = np.matmul(maxx.T, B).reshape((1, low_dim))  # maxx:1000*1  B:1000*6
        embedding_sample = np.append(embedding_sample, s, axis=0)
        # f_s = np.append(f_s, test_func.evaluate(maxx), axis=0 )
        f_s_true = np.append(f_s_true, test_func.evaluate_true(maxx), axis=0)

        # Standardize function values
        mean = f_s_true.mean()
        std = f_s_true.std()
        std = 1.0 if std < 1e-6 else std
        f = (f_s_true - mean) / std

        # Collecting data
        print("iter = ", i, "maxobj = ", np.max(f_s_true))
        best_results[0, i + initial_n] = np.max(f_s_true)

    return best_results, embedding_sample, f_s_true


def SIR(r, sample_x, sample_y, slice_number):
    nl = sample_x.shape[0]
    dim = sample_x.shape[1]
    xly = np.concatenate((sample_x, sample_y), axis=1)
    sort_xl = xly[np.argsort(xly[:, -1])][:, :-1]
    sample_mean = np.mean(sample_x, axis=0)

    sizeOfSlice = int(nl / slice_number)

    slice_mean = np.zeros((slice_number, dim))
    smean_c = np.zeros((slice_number, dim))
    # 计算每一个slice mean
    W = np.zeros((nl, slice_number))
    for i in range(slice_number):
        if i == slice_number - 1:
            for j in range(i * sizeOfSlice, nl):
                W[j][i] = 1
        else:
            for j in range(i * sizeOfSlice, (i + 1) * sizeOfSlice):
                W[j][i] = 1
    # print(W)
    # 解决下面的去generalized eigenvalue problem：Cov(HX)*V = lamda*Cov(X)*V
    cX = sample_x - np.tile(sample_mean, ((nl, 1)))
    Cov_X = np.matmul(cX.T, cX)
    WX = np.matmul(cX.T, W)
    Sigma_X = np.matmul(WX, WX.T)
    eigvals, eigvecs = eigh(Sigma_X, Cov_X + 0.01 * np.identity(dim), eigvals_only=False)
    B = eigvecs[:, dim - r:]
    return B


if __name__=='__main__':
    # from pyDOE import lhs
    # import random
    # s = lhs(100, 5) * 2 - 1
    # random.seed(0)
    # idx = random.sample([dim for dim in range(100)], 10)
    # idx.sort()
    # best_results, elapsed, embedding_sample, f_s_true = Run_Main(20, 100, 5, 20, 'ackley', s, idx)
    # print(best_results)
    import torch
    import datetime
    import random
    import time
    from pyDOE import lhs

    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.inf)

    dim_high = 100
    dim_low = 20
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
    path = "./results/" + func_name + "/SIR_BO_" + "D" + str(dim_high) + "_d" + str(dim_low) + "_de" + str(
        d_e) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    for seed in range(num_exp):
        random.seed(0)
        idx = random.sample([dim for dim in range(dim_high)], d_e)
        idx.sort()

        np.random.seed(seed)
        torch.manual_seed(seed)

        start = time.time()
        res, _, f_s_true = Run_Main(dim_low, dim_high, init_nums, budget - init_nums, func_name,
                                    lhs(dim_high, init_nums) * 2 - 1, idx)
        end = time.time()

        func_val_all[seed] = torch.from_numpy(res[0])
        fX = torch.from_numpy(f_s_true)
        # func_val_all_full[seed] *= fX.reshape(1, -1)[0, -1]
        # func_val_all_full[seed, 0:fX[0:budget, :].reshape(1, -1).shape[-1]] = fX[0:budget, :].reshape(1, -1)
        func_val_all_full[seed] = fX.reshape(1, -1)[0]
        time_all[seed] = torch.tensor(end - start)

        file_path = path + '/' + "seed" + str(seed) + "_" + datetime.datetime.now().strftime(
            '%m%d-%H-%M-%S').__str__() + ".txt"

        file = open(str(file_path), 'w')
        file.write("=============================== \n")
        file.write("EX: REMBO_Psi \n")
        file.write("Datetime: " + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
        file.write("=============================== \n\n\n")
        file.write("=============================== \n")
        file.write("          BASIC INFOS           \n")
        file.write("=============================== \n")
        file.write("D: " + str(dim_high) + " \n")
        file.write("d: " + str(dim_low) + " \n")
        file.write("d_e: " + str(d_e) + " \n")
        file.write("Effective dim:" + str(idx) + " \n")
        file.write("Init points: " + str(init_nums) + " \n")
        # file.write("x*:" + str(x_best) + "\n")
        file.write("optimal value:" + str(func_val_all[seed, -1]) + "\n")
        file.write("Total time consume: " + str(end - start) + " s \n")
        file.write("=============================== \n\n\n")

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

