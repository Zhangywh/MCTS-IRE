import GPy
# import matlab.engine
import numpy as np
import math
from pyDOE import lhs
from scipy.stats import norm
import functions
import projection_matrix
import projections
import kernel_inputs
import timeit
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def EI(D_size,f_max,mu,var):
    """
    :param D_size: number of points for which EI function will be calculated
    :param f_max: the best value found for the test function so far
    :param mu: a vector of predicted values for mean of the test function
        corresponding to the points
    :param var: a vector of predicted values for variance of the test function
        corresponding to the points
    :return: a vector of EI values of the points
    """
    ei=np.zeros((D_size,1))
    std_dev=np.sqrt(var)
    for i in range(D_size):
        if var[i]!=0:
            z= (mu[i] - f_max) / std_dev[i]
            ei[i]= (mu[i]-f_max) * norm.cdf(z) + std_dev[i] * norm.pdf(z)
    return ei


def RunRembo(low_dim=2, high_dim=20, initial_n=20, total_itr=100, func_type='Branin',
             matrix_type='simple', kern_inp_type='Y', A_input=None, s=None, active_var=None,
             hyper_opt_interval=20, ARD=False, variance=1., length_scale=None, box_size=None,
             noise_var=0):
    """"

    :param low_dim: the dimension of low dimensional search space
    :param high_dim: the dimension of high dimensional search space
    :param initial_n: the number of initial points
    :param total_itr: the number of iterations of algorithm. The total
        number of test function evaluations is initial_n + total_itr
    :param func_type: the name of test function
    :param matrix_type: the type of projection matrix
    :param kern_inp_type: the type of projection. Projected points
        are used as the input of kernel
    :param A_input: a projection matrix with iid gaussian elements.
        The size of matrix is low_dim * high_dim
    :param s: initial points
    :param active_var: a vector with the size of greater or equal to
        the number of active variables of test function. The values of
        vector are integers less than high_dim value.
    :param hyper_opt_interval: the number of iterations between two consecutive
        hyper parameters optimizations
    :param ARD: if TRUE, kernel is isomorphic
    :param variance: signal variance of the kernel
    :param length_scale: length scale values of the kernel
    :param box_size: this variable indicates the search space [-box_size, box_size]^d
    :param noise_var: noise variance of the test functions
    :return: a tuple of best values of each iteration, all observed points, and
        corresponding test function values of observed points
    """

    if active_var is None:
        active_var = np.arange(high_dim)
    if box_size is None:
        box_size = math.sqrt(low_dim)
    if hyper_opt_interval is None:
        hyper_opt_interval = 10

    #Specifying the type of objective function
    if func_type == 'Branin':
        test_func = functions.Branin(active_var, noise_var=noise_var)
    elif func_type == 'Hartmann6':
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

    #Specifying the type of embedding matrix
    if matrix_type == 'simple':
        matrix = projection_matrix.SimpleGaussian(low_dim, high_dim)
    elif matrix_type == 'normal':
        matrix = projection_matrix.Normalized(low_dim, high_dim)
    elif matrix_type == 'orthogonal':
        matrix = projection_matrix.Orthogonalized(low_dim, high_dim)
    else:
        TypeError('The input for matrix_type variable is invalid, which is', matrix_type)
        return

    # Generating matrix A
    if A_input is not None:
        matrix.A = A_input

    A = matrix.evaluate()

    #Specifying the input type of kernel
    if kern_inp_type == 'Y':
        kern_inp = kernel_inputs.InputY(A)
        input_dim = low_dim
    elif kern_inp_type == 'X':
        kern_inp = kernel_inputs.InputX(A)
        input_dim = high_dim
    elif kern_inp_type == 'psi':
        kern_inp = kernel_inputs.InputPsi(A)
        input_dim = high_dim
    else:
        TypeError('The input for kern_inp_type variable is invalid, which is', kern_inp_type)
        return

    #Specifying the convex projection
    cnv_prj=projections.ConvexProjection(A)

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed = np.zeros([1, total_itr + initial_n])

    # Initiating first sample    # Sample points are in [-d^1/2, d^1/2]
    if s is None:
        s = lhs(low_dim, initial_n) * 2 * box_size - box_size
    # f_s = test_func.evaluate(cnv_prj.evaluate(s))
    f_s_true = test_func.evaluate_true(cnv_prj.evaluate(s))
    for i in range(initial_n):
        best_results[0, i] = np.max(f_s_true[0:i+1])

    # Generating GP model
    k = GPy.kern.Matern52(input_dim=input_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(kern_inp.evaluate(s), f_s_true, kernel=k)
    m.likelihood.variance = 1e-6

    # Main loop of the algorithm
    for i in range(total_itr):

        start = timeit.default_timer()
        # Updating GP model
        m.set_XY(kern_inp.evaluate(s), f_s_true)
        if (i+initial_n<=25 and i % 5 == 0) or (i+initial_n>25 and i % hyper_opt_interval == 0):
            m.optimize()

        # finding the next point for sampling
        D = lhs(low_dim, 2000) * 2 * box_size - box_size
        mu, var = m.predict(kern_inp.evaluate(D))
        ei_d = EI(len(D), max(f_s_true), mu, var)
        index = np.argmax(ei_d)
        s = np.append(s, [D[index]], axis=0)
        # f_s = np.append(f_s, test_func.evaluate(cnv_prj.evaluate([D[index]])), axis=0)
        f_s_true = np.append(f_s_true, test_func.evaluate_true(cnv_prj.evaluate([D[index]])), axis=0)
        print('iteration ', i + 1, 'f(x)=', f_s_true[-1])

        #Collecting data
        stop = timeit.default_timer()
        best_results[0, i + initial_n] = np.max(f_s_true)
        elapsed[0, i + initial_n] = stop - start

    # if func_type == 'WalkerSpeed':
    #     eng.quit()

    return best_results, elapsed, s, f_s_true, cnv_prj.evaluate(s)


if __name__=='__main__':
    # res,_, s, fs_true, high_s = RunRembo(low_dim=2, high_dim=100, func_type='ackley', initial_n=10,
    #                                          total_itr=50, kern_inp_type='psi', active_var=[i for i in range(10)], ARD=True, noise_var=1)
    # print(res, fs_true)
    import torch
    import datetime
    import random
    import time

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
    path = "./results/" + func_name + "/REMBO_Psi_" + "D" + str(dim_high) + "_d" + str(dim_low) + "_de" + str(
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
        res, _, s, f_s_true, _ = RunRembo(low_dim=dim_low, high_dim=dim_high, initial_n=init_nums,
                                          total_itr=budget - init_nums, func_type=func_name, kern_inp_type='psi',
                                          active_var=idx, ARD=True, noise_var=0)
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


