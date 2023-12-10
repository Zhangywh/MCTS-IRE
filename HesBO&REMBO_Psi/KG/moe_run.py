from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
import os, sys
import time
from embd_functions import *
import pickle

from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import PosteriorMeanMCMC
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import GaussianProcessLogLikelihoodMCMC as cppGaussianProcessLogLikelihoodMCMC
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import posterior_mean_optimization, PosteriorMean

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
from moe.optimal_learning.python.default_priors import DefaultPrior

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters as pyGradientDescentParameters
from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer as pyGradientDescentOptimizer
from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

from examples import bayesian_optimization

# arguments for calling this script:
# python main.py [obj_func_name] [method_name] [num_to_sample] [job_id]
# example: python main.py Branin KG 4 1
# you can define your own obj_function and then just change the objective_func object below, and run this script.

argv = sys.argv[1:]
obj_func_name = str(argv[0])
method = str(argv[1])
num_func_eval = int(argv[2])
low_dim = int(argv[3])
high_dim = int(argv[4])
initial_n = int(argv[5])
job_id = int(argv[6])
if len(argv)<8:
    noise_var=0
    noisy = False
else:
    noise_var=int(argv[7])
    noisy = True

# constants
num_opt_start=12
num_to_sample = 1
num_iteration = int(old_div(num_func_eval, num_to_sample)) + 1
act_var=np.arange(high_dim)
high_to_low=np.random.choice(range(low_dim), high_dim)
sign = np.random.choice([-1, 1], high_dim)
bx_size=1
file_name='moe_results/'+obj_func_name+'_MOE_d'+str(low_dim)+'_D'+str(high_dim)+'_init_'+str(initial_n)+'_rep_'+str(job_id)
data_str = ''

obj_func_dict = {'Branin': Branin(act_var, low_dim, high_to_low, sign, bx_size, noise_var= noise_var),
                 'Rosenbrock': Rosenbrock(act_var, low_dim, high_to_low, sign, bx_size, noise_var= noise_var),
                 'Hartmann6': Hartmann6(act_var, low_dim, high_to_low, sign, bx_size, noise_var= noise_var),
                 'StybTang': StybTang(act_var, low_dim, high_to_low, sign, bx_size, noise_var= noise_var),
                 'MNIST': MNIST(act_var, low_dim, high_to_low, sign, bx_size)}

objective_func = obj_func_dict[obj_func_name]
dim = int(objective_func._dim)
num_initial_points = initial_n

num_fidelity = objective_func._num_fidelity

inner_search_domain = pythonTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
                                                 for i in range(objective_func._search_domain.shape[0]-num_fidelity)])
cpp_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])
cpp_inner_search_domain = cppTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
                                                  for i in range(objective_func._search_domain.shape[0]-num_fidelity)])

# get the initial data
init_pts = np.zeros((initial_n, objective_func._dim))
init_pts[:, :objective_func._dim-objective_func._num_fidelity] = inner_search_domain.generate_uniform_random_points_in_domain(initial_n)
for pt in init_pts:
    pt[objective_func._dim-objective_func._num_fidelity:] = np.ones(objective_func._num_fidelity)

# observe
derivatives = objective_func._observations
observations = [0] + [i+1 for i in derivatives]
init_pts_value = np.array([objective_func.evaluate(pt) for pt in init_pts])#[:, observations]
true_value_init = np.array([objective_func.evaluate_true(pt) for pt in init_pts])#[:, observations]

# Collecting Data
s_suggest=np.array(init_pts)
f_s_suggest=np.array(init_pts_value).reshape(initial_n, 1)
s_recommend=np.array(init_pts)
f_s_recommend=np.array(true_value_init).reshape(initial_n, 1)
elapsed=np.zeros([1, num_iteration + initial_n])

init_data = HistoricalData(dim = objective_func._dim, num_derivatives = len(derivatives))
init_data.append_sample_points([SamplePoint(pt, [init_pts_value[num, i] for i in observations],
                                            objective_func._sample_var) for num, pt in enumerate(init_pts)])

# initialize the model
prior = DefaultPrior(1+dim+len(observations), len(observations))

# noisy = False means the underlying function being optimized is noise-free
cpp_gp_loglikelihood = cppGaussianProcessLogLikelihoodMCMC(historical_data = init_data,
                                                           derivatives = derivatives,
                                                           prior = prior,
                                                           chain_length = 1000,
                                                           burnin_steps = 2000,
                                                           n_hypers = 2 ** 4,
                                                           noisy = noisy)
cpp_gp_loglikelihood.train()

py_sgd_params_ps = pyGradientDescentParameters(max_num_steps=1000,
                                               max_num_restarts=3,
                                               num_steps_averaged=15,
                                               gamma=0.7,
                                               pre_mult=1.0,
                                               max_relative_change=0.02,
                                               tolerance=1.0e-10)

cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1,
                                                 max_num_steps=6,
                                                 max_num_restarts=1,
                                                 num_steps_averaged=3,
                                                 gamma=0.0,
                                                 pre_mult=1.0,
                                                 max_relative_change=0.1,
                                                 tolerance=1.0e-10)

cpp_sgd_params_kg = cppGradientDescentParameters(num_multistarts=200,
                                                 max_num_steps=50,
                                                 max_num_restarts=2,
                                                 num_steps_averaged=4,
                                                 gamma=0.7,
                                                 pre_mult=1.0,
                                                 max_relative_change=0.5,
                                                 tolerance=1.0e-10)

# minimum of the mean surface
eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e3))
eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                      (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled,
                       cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

test = np.zeros(eval_pts.shape[0])
ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models, num_fidelity)
for i, pt in enumerate(eval_pts):
    ps.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
    test[i] = -ps.compute_objective_function()
report_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
report_point = multistart_optimize(ps_mean_opt, report_point, num_multistarts = 1)[0]
report_point = report_point.ravel()
report_point = np.concatenate((report_point, np.ones(objective_func._num_fidelity)))

print("best so far in the initial data {0}".format(true_value_init[np.argmin(true_value_init[:,0])][0]))
capital_so_far = 0.
start=time.time()
for n in range(num_iteration):
    print(method + ", {0}th job, {1}th iteration, func={2}, q={3}".format(
            job_id, n, obj_func_name, num_to_sample
    ))
    time1 = time.time()
    if method == 'KG':
        discrete_pts_list = []

        discrete, _ = bayesian_optimization.gen_sample_from_qei_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_search_domain,
                                                                cpp_sgd_params_kg, 10, num_mc=2 ** 10)
        for i, cpp_gp in enumerate(cpp_gp_loglikelihood.models):
            discrete_pts_optima = np.array(discrete)

            eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e3))
            eval_pts = np.reshape(np.append(eval_pts,
                                            (cpp_gp.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                                  (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim-objective_func._num_fidelity))

            test = np.zeros(eval_pts.shape[0])
            ps_evaluator = PosteriorMean(cpp_gp, num_fidelity)
            for i, pt in enumerate(eval_pts):
                ps_evaluator.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
                test[i] = -ps_evaluator.compute_objective_function()

            initial_point = eval_pts[np.argmin(test)]

            ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)
            report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess = initial_point, max_num_threads = 4)

            ps_evaluator.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            if -ps_evaluator.compute_objective_function() > np.min(test):
                report_point = initial_point

            discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
                                             (discrete_pts_optima.shape[0] + 1, cpp_gp.dim-objective_func._num_fidelity))
            discrete_pts_list.append(discrete_pts_optima)

        ps_evaluator = PosteriorMean(cpp_gp_loglikelihood.models[0], num_fidelity)
        ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)
        # KG method
        next_points, voi = bayesian_optimization.gen_sample_from_qkg_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_gp_loglikelihood.models,
                                                                ps_sgd_optimizer, cpp_search_domain, num_fidelity, discrete_pts_list,
                                                                cpp_sgd_params_kg, num_to_sample, num_mc=2 ** 7)

    elif method == 'EI':
        # EI method
        next_points, voi = bayesian_optimization.gen_sample_from_qei(cpp_gp_loglikelihood.models[0], cpp_search_domain,
                                                           cpp_sgd_params_kg, num_to_sample, num_mc=2 ** 10)
    else:
        print(method + str(" not supported"))
        sys.exit(0)

    print(method + " takes "+str((time.time()-time1))+" seconds")
    #time1 = time.time()
    print(method + " suggests points:")
    print(next_points)

    sampled_points = [SamplePoint(pt, objective_func.evaluate(pt)[observations], objective_func._sample_var) for pt in next_points]

    #print "evaluating takes "+str((time.time()-time1)/60)+" mins"
    capitals = np.ones(num_to_sample)
    for i in range(num_to_sample):
        if num_fidelity > 0:
            value = 1.0
            for j in range(num_fidelity):
                value *= next_points[i, dim-1-j]
            capitals[i] = value
    capital_so_far += np.amax(capitals)
    print("evaluating takes capital " + str(capital_so_far) +" so far")

    # retrain the model
    time1 = time.time()

    cpp_gp_loglikelihood.add_sampled_points(sampled_points)
    cpp_gp_loglikelihood.train()

    print("retraining the model takes "+str((time.time()-time1))+" seconds")
    time1 = time.time()

    # report the point
    if method == 'KG':
        eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e4))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models, num_fidelity)
        test = np.zeros(eval_pts.shape[0])
        for i, pt in enumerate(eval_pts):
            ps.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            test[i] = -ps.compute_objective_function()
        # initial_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))
        top_k = np.argsort(test)[:num_opt_start]
        initial_point = np.reshape([eval_pts[t] for t in top_k],
                                   (num_opt_start,cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
        ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
        report_point = multistart_optimize(ps_mean_opt, initial_point, num_multistarts=num_opt_start)[0]

        ps.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
        if -ps.compute_objective_function() > np.min(test):
            report_point = initial_point[0].reshape((1, cpp_gp_loglikelihood.dim - objective_func._num_fidelity))
    else:
        cpp_gp = cpp_gp_loglikelihood.models[0]
        report_point = (cpp_gp.get_historical_data_copy()).points_sampled[np.argmin(cpp_gp._points_sampled_value[:, 0])]

    report_point = report_point.ravel()
    report_point = np.concatenate((report_point, np.ones(objective_func._num_fidelity)))

    print()
    print("Optimization finished successfully!")
    print("The recommended point: ", end=' ')
    print(report_point)
    vall=objective_func.evaluate_true(report_point)[0]
    print("recommending the point takes "+str((time.time()-time1))+" seconds")
    print(method + ", VOI {0}, best so far {1}".format(voi, vall[0]))

    s_suggest=np.append(s_suggest, [sampled_points[0][0]], axis=0)
    f_s_suggest=np.append(f_s_suggest, sampled_points[0][1], axis=0)
    s_recommend=np.append(s_recommend, [report_point], axis=0)
    f_s_recommend = np.append(f_s_recommend, [vall], axis=0)
    elapsed[0, n + initial_n] = time.time() - start
    pickle.dump([f_s_suggest, elapsed, s_suggest, s_recommend, f_s_recommend, high_to_low, sign], open(file_name, 'wb'))
    # Collecting the logfile
    logfile = open(file_name + '.out', 'w')
    data_str = data_str + 'Iter: ' + str(n) + '------------------------------' + '\n'
    data_str = data_str + 'Top_k values of mean posterior:' + str([test[t] for t in top_k]) + '\n'
    data_str = data_str + 'True values of suggested points:' + str(objective_func.evaluate_true(s_suggest)) + '\n'
    data_str = data_str + 'Noisy values of suggested points:' + str(f_s_suggest) + '\n'
    data_str = data_str + 'True values of recommended points:' + str(f_s_recommend) + '\n'
    logfile.writelines(data_str)
    logfile.close()


