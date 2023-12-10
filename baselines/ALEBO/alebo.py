import numpy as np
import torch
from core import ObjFunc
import time
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.service.managed_loop import optimize
from matplotlib import pyplot as plt
import random
import os
import datetime

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)


dim_high = 100
dim_low = 20
d_e = 10
seed = 0
budget = 300
num_exp = 5
init_nums = 5
func_name = 'ackley'
target = ObjFunc.ackley_rand
# bounds = [[-5., 10.]] * dim_high
bounds = [[-32.768, 32.768]] * dim_high
# bounds = [[-5.12, 5.12]] * dim_high
# bounds = [[-50., 50.]] * dim_high
# bounds = torch.tensor([[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05,
#                         1.05, 1.15, 1.7, -torch.pi, -torch.pi, -torch.pi, -torch.pi],
#                        [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5,
#                         291.0, torch.pi, torch.pi, torch.pi, torch.pi]]).T.tolist()
func_val_all = torch.zeros(num_exp, budget)
func_val_all_full = torch.ones(num_exp, budget)
time_all = torch.zeros(num_exp)

folder = os.path.exists("../results")
if not folder:
    os.makedirs("../results")
path = "./results/" + func_name + "/ALEBO_" + "D" + str(dim_high) + "_d" + str(dim_low) + "_de" + str(d_e) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)


for seed in range(num_exp):
    random.seed(0)
    idx = random.sample([dim for dim in range(dim_high)], d_e)
    idx.sort()
    parameters = [{"name": f"x{i}", "type": "range", "bounds": bounds[i], "value_type": "float"}
                  for i in range(dim_high)]

    def obj_func(parameterization):
        x = torch.tensor([[parameterization[f'x{i}'] for i in range(dim_high)]])
        res = -target(x, idx)
        return {"objective": (res.numpy(), 0.0)}
    np.random.seed(seed)
    torch.manual_seed(seed)

    alebo_strategy = ALEBOStrategy(D=dim_high, d=dim_low, init_size=init_nums)
    start = time.time()
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        experiment_name=func_name,
        objective_name="objective",
        evaluation_function=obj_func,
        minimize=True,
        total_trials=budget,
        generation_strategy=alebo_strategy,
    )
    end = time.time()

    objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])  # Observed values
    # print(objectives)



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
    func_val_all[seed] = -torch.from_numpy(np.minimum.accumulate(objectives))
    fX = -torch.from_numpy(objectives)
    # func_val_all_full[seed] *= fX.reshape(1, -1)[0, -1]
    # func_val_all_full[seed, 0:fX[0:budget, :].reshape(1, -1).shape[-1]] = fX[0:budget, :].reshape(1, -1)
    func_val_all_full[seed] = fX
    time_all[seed] = torch.tensor(end - start)

    file_path = path + '/' + "seed" + str(seed) + "_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__() + ".txt"

    file = open(str(file_path), 'w')
    file.write("=============================== \n")
    file.write("EX: ALEBO \n")
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

# torch.save(torch.from_numpy(X), path + '/x.pt')
# torch.save(torch.from_numpy(fX), path + '/f.pt')







# parameters = [{"name": f"x{i}", "type": "range", "bounds": [-1., 1.], "value_type": "float"}
#               for i in range(100)]
#
# alebo_strategy = ALEBOStrategy(D=100, d=20, init_size=10)
# start = time.time()
# best_parameters, values, experiment, model = optimize(
#     parameters=parameters,
#     experiment_name="hartmann6",
#     objective_name="objective",
#     evaluation_function=obj_func,
#     minimize=True,
#     total_trials=200,
#     generation_strategy=alebo_strategy,
# )
# end = time.time()
# objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
# print(objectives)
# print('total time', end-start)
#
# fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(111)
# ax.grid(alpha=0.2)
# ax.plot(range(1, 301), np.minimum.accumulate(objectives))
# # ax.axhline(y=branin.fmin, ls='--', c='k')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Best objective found')
# plt.savefig(fname='hartmann6.svg', format="svg", dpi=1200)
# plt.show()

# res = [22.97311974 24.27016449 25.08746719 25.12780952 24.69402695 24.03247452
#  24.08364296 25.61862564 24.86052132 25.8758049  25.13498878 26.32372475
#  25.56519508 26.02170563 24.37950706 24.30269623 24.97946167 24.45555496
#  25.61425781 26.08221245 25.01789856 23.36754799 24.28687096 24.45039749
#  24.83372116 25.70624924 26.06266594 24.4370327  24.7278614  25.13919258
#  25.82083321 25.33169746 24.53774643 24.18442917 24.4651947  25.2312603
#  25.79554367 23.21154594 25.05375671 26.16346931 27.05702591 24.26603699
#  23.63068962 25.95162582 25.34884453 24.29056931 24.8746109  25.22290802
#  23.78292274 24.8419075  22.74346161 24.55327034 24.73771477 26.14715004
#  26.21533203 23.06640244 25.46063423 26.23371315 24.99298668 23.27514648
#  25.8030529  26.20947647 21.58095169 25.14548111 23.94132042 25.89892387
#  25.79453468 24.02658081 25.97193909 25.48855019 25.0369072  24.79569626
#  24.45955467 24.9983902  23.68529129 24.52689362 24.64606667 24.42663956
#  24.78530693 27.43220711 25.08740807 25.34477615 23.64751434 27.42948151
#  23.83962631 24.39255905 24.87654114 25.47050285 26.22628403 24.16123962
#  23.55195427 27.41365433 27.06155586 27.78601265 24.43372154 26.891819
#  27.93790436 27.21258163 23.67766762 24.1626339 ]

# hartmann_res = [-0.06967802 -0.17587954 -0.02825422 -0.4816747  -1.01503038 -0.34022698
# -0.37832293 -0.17264855 -0.15394044 -0.03529463 -0.17950743 -0.54487687
# -0.18563066 -0.41602397 -0.53131652 -0.02647183 -0.77501673 -0.89066511
# -1.03287351 -0.68584061 -0.75034463 -0.34147909 -0.68399161 -0.22920734
# -0.11633577 -0.30479488 -0.21229637 -0.96884412 -0.42441568 -0.4617157
# -0.8214336  -0.82587051 -0.29007369 -0.83198625 -0.88847041 -0.28200272
# -0.63194185 -0.57756549 -0.88277358 -0.52081412 -0.06070245 -0.44436193
# -0.06896272 -1.42369413 -0.67941368 -1.56874955 -0.98749179 -2.32100296
# -0.22312516 -1.97910774 -0.175864   -0.07323027 -0.05481159 -0.47992277
# -0.42992672 -1.01378036 -0.56996787 -0.2148007  -0.32660174 -0.29674578
# -1.3119719  -0.15463369 -1.16696978 -0.08056363 -0.00725569 -0.33639151
# -0.06137242 -0.505247   -0.95763695 -0.53325766 -0.49188474 -0.77318692
# -0.18451157 -0.99605936 -1.11012232 -0.39915028 -0.33793169 -1.66895187
# -0.08420289 -0.30684084 -0.69610012 -0.22044396 -0.7616176  -0.91307592
# -0.32165343 -0.58042848 -0.20751628 -1.08467674 -1.72338974 -0.85417014
# -0.0533395  -0.6650992  -0.18680038 -1.20765305 -0.12790623 -1.04114342
# -0.27349296 -1.70761132 -0.26084483 -0.79330564 -0.66827077 -0.15456548
# -0.88445473 -1.09774435 -0.24314466 -1.36014473 -0.08684492 -1.40197039
# -0.06512143 -1.49338925 -0.68130404 -1.57289302 -0.10736579 -0.80046695
# -1.22730768 -0.03662468 -0.33800644 -0.05784395 -1.71326876 -0.25089616
# -0.58620858 -2.14128828 -1.49334288 -0.05850459 -1.33425641 -0.66337943
# -0.51031691 -2.43046522 -2.83209324 -0.2996093  -1.26538765 -0.84663731
# -0.06895436 -1.01439309 -1.34195209 -0.67720413 -0.10530256 -0.05417699
# -0.02368342 -0.8294009  -0.01118531 -1.11026597 -0.41029221 -0.51275766
# -1.67127609 -0.7604109  -0.24941327 -0.46552029 -0.96185189 -1.35230458
# -1.41998672 -0.41328764 -0.33919919 -0.07749743 -1.39017928 -0.26713431
# -0.51433313 -0.02574678 -0.71074384 -0.21137926 -1.39323533 -0.71507472
# -0.43027401 -0.51407051 -1.48875618 -2.81369257 -0.17780726 -0.15558133
# -0.1123537  -1.55932355 -0.85335916 -0.58244234 -0.20508313 -1.63972354
# -0.06395398 -0.52044499 -2.32462358 -0.21859382 -0.46720329 -0.0374506
# -0.00381689 -1.44964075 -0.37029698 -0.01677367 -0.58122617 -0.02487818
# -1.27346241 -0.76297057 -0.10974891 -1.22364235 -0.62581545 -1.2608248
# -0.70490444 -1.9226352  -1.83625317 -0.60856611 -2.0996654  -0.50465143
# -0.01507665 -0.05739002]
