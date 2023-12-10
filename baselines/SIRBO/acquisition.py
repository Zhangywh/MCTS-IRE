import numpy as np
from scipy.stats import norm
from pyDOE import lhs


class ACfunction:
    def __init__(self, B, GP, initial_size, low_dimension):
        self.B = B
        self.GP = GP
        self.n = initial_size
        self.d = low_dimension
        self.embedding_sample = 0
        self.D = 0
        self.flag = True
        self.fs_true = 0

    def update(self, B, GP, initial_size, low_dimension):
        self.B = B
        self.GP = GP
        self.n = initial_size
        self.d = low_dimension

    def updateQP(self, embedding_sample):
        self.embedding_sample = embedding_sample

    def acfunctionUCB(self, X):
        X = X.reshape((1, len(X)))
        Y = np.matmul(X, self.B)
        # print(X, Y)
        mu, var = self.GP.predict(Y)
        # print(mu, var)
        kappa = np.sqrt(2 * np.log(self.n ** (self.d / 2 + 2) * np.pi ** 2 / (3 * 0.1)))
        ucb_d = self.UCB(mu, var, kappa=kappa).reshape(1)
        return ucb_d

    def originalUCB(self, mu, var):
        kappa = np.sqrt(2 * np.log(self.n ** (self.d / 2 + 2) * np.pi ** 2 / (3 * 0.1)))
        ucb_d = self.UCB(mu, var, kappa=kappa)
        return ucb_d

    def newfunction(self, X):
        X = X.reshape((1, len(X)))
        Y = np.matmul(X, self.B)
        box = np.sum(self.B, axis=1)
        if (self.flag):
            self.D = lhs(self.d, 2000) * 2 * np.max(np.abs(box)) - np.max(np.abs(box))
            self.flag = False
        mu, var = self.GP.predict(self.D)
        kappa = np.sqrt(2 * np.log(self.n ** (self.d / 2 + 2) * np.pi ** 2 / (3 * 0.1)))
        # ucb
        ucb_d = self.UCB(mu, var, kappa=kappa)
        index = np.argmax(ucb_d)

        # ei
        # ei_d = self.acfunctionEI(mu,var)
        # index = np.argmax(ei_d)

        aa = (Y - self.D[index])
        bb = np.inner(aa, aa).reshape(-1)
        return bb

    def resetflag(self, Bnew):
        self.flag = True
        self.B = Bnew

    def UCB(self, mu, var, kappa=0.01):
        return (mu + kappa * var)

    def set_fs_true(self, fs_true):
        self.fs_true = fs_true

    def acfunctionEI(self, mu, var):
        """
        :param D_size: number of points for which EI function will be calculated
        :param f_max: the best value found for the test function so far
        :param mu: a vector of predicted values for mean of the test function
            corresponding to the points
        :param var: a vector of predicted values for variance of the test function
            corresponding to the points
        :return: a vector of EI values of the points
        """
        # box = np.sum(self.B, axis=1)
        #
        # D = lhs(dim, 2000) * 2 * np.max(np.abs(box)) - np.max(np.abs(box))
        f_max = self.fs_true
        # mu,var = self.GP.predict(D)
        D_size = self.D.shape[0]
        ei = np.zeros((D_size, 1))
        std_dev = np.sqrt(var)
        for i in range(D_size):
            if var[i] != 0:
                z = (mu[i] - f_max) / std_dev[i]
                ei[i] = (mu[i] - f_max) * norm.cdf(z) + std_dev[i] * norm.pdf(z)
        # index = np.argmax(ei)
        # unlabeled data
        # idx = np.argsort(np.array(-ei), axis=0).reshape(-1)[:self.n]
        # xu = D[idx]
        return ei

    def qp(self, x):
        # G = np.eye(high_dim)
        # h = np.ones((high_dim, 1)).reshape(-1)
        x = np.array(x).reshape((len(x), 1))
        p = np.matmul(self.B.T, self.B)
        q = -np.matmul(self.B.T, np.array(self.embedding_sample).reshape((self.d, 1)))
        # A = np.zeros((low_dim, high_dim))
        # b = np.zeros((low_dim, 1)).reshape(-1)
        Y = np.matmul(np.matmul(x.T, p), x) + 2 * np.matmul(q.T, x)
        return Y.reshape(-1)
