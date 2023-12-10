import numpy as np

f = open('results/ackley/LaMCTS_D100_de10_0507-14-25-22/result1000')
res = np.zeros([8, 1000])
for i in range(8):
    a = f.readline()
    print(a, type(a))
    a = [float(i) for i in a[1: -2].split(', ')]
    # print(a, type(a), len(a), type(a[0]))
    a = np.array(a)
    a = np.minimum.accumulate(a)
    res[i] = a
f.close()
print(res.mean(axis=0).tolist())
print(res.std(axis=0).tolist())
