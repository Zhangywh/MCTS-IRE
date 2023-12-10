import GPy
# import matlab.engine
import numpy as np
from pyDOE import lhs
import functions
from rembo_psi import EI
import timeit
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def dim_sampling(low_dim, X, bx_size):
    if len(X.shape)==1:
        X=X.reshape((1, X.shape[0]))
    n=X.shape[0]
    high_dim=X.shape[1]
    low_obs=np.zeros((n,low_dim))
    high_to_low=np.zeros(high_dim,dtype=int)
    sign=np.random.choice([-1,1],high_dim)
    for i in range(high_dim):
        high_to_low[i]=np.random.choice(range(low_dim))
        low_obs[:,high_to_low[i]]=X[:,i]*sign[i]+ low_obs[:,high_to_low[i]]

    for i in range(n):
        for j in range(low_dim):
            if low_obs[i][j] > bx_size: low_obs[i][j] = bx_size
            elif low_obs[i][j] < -bx_size: low_obs[i][j] = -bx_size
    return low_obs, high_to_low, sign

def back_projection(low_obs, high_to_low, sign, bx_size):
    if len(low_obs.shape)==1:
        low_obs=low_obs.reshape((1, low_obs.shape[0]))
    n=low_obs.shape[0]
    high_dim=high_to_low.shape[0]
    low_dim=low_obs.shape[1]
    high_obs=np.zeros((n,high_dim))
    scale=1
    for i in range(high_dim):
        high_obs[:,i]=sign[i]*low_obs[:,high_to_low[i]]*scale
    for i in range(n):
        for j in range(high_dim):
            if high_obs[i][j] > bx_size: high_obs[i][j] = bx_size
            elif high_obs[i][j] < -bx_size: high_obs[i][j] = -bx_size
    return high_obs

def RunMain(low_dim=2, high_dim=25, initial_n=20, total_itr=100, func_type='Branin',
            s=None, active_var=None, ARD=False, variance=1., length_scale=None, box_size=None,
            high_to_low=None, sign=None, hyper_opt_interval=20, noise_var=0):
    """

    :param high_dim: the dimension of high dimensional search space
    :param low_dim: The effective dimension of the algorithm.
    :param initial_n: the number of initial points
    :param total_itr: the number of iterations of algorithm. The total
        number of test function evaluations is initial_n + total_itr
    :param func_type: the name of test function
    :param s: initial points
    :param active_var: a vector with the size of greater or equal to
        the number of active variables of test function. The values of
        vector are integers less than high_dim value.
    :param ARD: if TRUE, kernel is isomorphic
    :param variance: signal variance of the kernel
    :param length_scale: length scale values of the kernel
    :param box_size: this variable indicates the search space [-box_size, box_size]^d
    :param high_to_low: a vector with D elements. each element can have a value from {0,..,d-1}
    :param sign: a vector with D elements. each element is either +1 or -1.
    :param hyper_opt_interval: the number of iterations between two consecutive
        hyper parameters optimizations
    :param noise_var: noise variance of the test functions
    :return: a tuple of best values of each iteration, all observed points, and
        corresponding test function values of observed points
    """

    if active_var is None:
        active_var = np.arange(high_dim)
    if box_size is None:
        box_size = 1
    if high_to_low is None:
        high_to_low = np.random.choice(range(low_dim), high_dim)
    if sign is None:
        sign = np.random.choice([-1, 1], high_dim)

    #Specifying the type of objective function
    if func_type=='Branin':
        test_func = functions.Branin(active_var, noise_var=noise_var)
    elif func_type=='Hartmann6':
        test_func = functions.Hartmann6(active_var, noise_var=noise_var)
    elif func_type == 'StybTang':
        test_func = functions.StybTang(active_var, noise_var=noise_var)
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
    elapsed=np.zeros([1, total_itr + initial_n])

    # Creating the initial points. The shape of s is nxD
    if s is None:
        s=lhs(low_dim, initial_n) * 2 * box_size - box_size
    # f_s = test_func.evaluate(back_projection(s,high_to_low,sign,box_size))
    f_s_true = test_func.evaluate_true(back_projection(s,high_to_low,sign,box_size))
    for i in range(initial_n):
        best_results[0,i]=np.max(f_s_true[0:i+1])

    # Building and fitting a new GP model
    kern = GPy.kern.Matern52(input_dim=low_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(s, f_s_true, kernel=kern)
    m.likelihood.variance = 1e-3
    # print(s, f_s_true)

    # Main loop
    for i in range(total_itr):

        start = timeit.default_timer()

        # Updating GP model
        m.set_XY(s, f_s_true)
        if (i+initial_n<=25 and i % 5 == 0) or (i+initial_n>25 and i % hyper_opt_interval == 0):
            m.optimize()

        # Maximizing acquisition function
        D = lhs(low_dim, 2000) * 2 * box_size - box_size
        mu, var = m.predict(D)
        ei_d = EI(len(D), max(f_s_true), mu, var)
        index = np.argmax(ei_d)

        # Adding the new point to our sample
        s = np.append(s, [D[index]], axis=0)
        new_high_point = back_projection(D[index],high_to_low,sign,box_size)
        # f_s = np.append(f_s, test_func.evaluate(new_high_point), axis=0)
        f_s_true = np.append(f_s_true, test_func.evaluate_true(new_high_point), axis=0)

        print('iteration ', i + 1, 'f(x)=', f_s_true[-1])

        stop = timeit.default_timer()
        best_results[0, i + initial_n] = np.max(f_s_true)
        elapsed[0, i + initial_n]=stop-start

    # if func_type == 'WalkerSpeed':
    #     eng.quit()
    high_s = back_projection(s,high_to_low,sign,box_size)
    return best_results, elapsed, s, f_s_true, high_s

if __name__=='__main__':
    import torch
    import datetime
    import random
    import time


    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.inf)

    dim_high = 100
    dim_low = 20
    d_e = 20
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
    path = "./results/" + func_name + "/HesBO_" + "D" + str(dim_high) + "_d" + str(dim_low) + "_de" + str(
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
        res, _, s, f_s_true, _ = RunMain(low_dim=dim_low, high_dim=dim_high, initial_n=init_nums,
                                         total_itr=budget - init_nums, func_type=func_name, active_var=idx,
                                         ARD=True, noise_var=0)

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
        file.write("EX: HesBO \n")
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
    # res, time, s, f_s_true, _ = RunMain(low_dim=10, high_dim=100, initial_n=20, total_itr=50, func_type='Rosenbrock', active_var=[i for i in range(10)], ARD=True, noise_var=0)
    # print(res)
