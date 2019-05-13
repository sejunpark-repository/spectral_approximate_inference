import numpy as np
from gm.ising_model import IsingModel
from read_mat import ReadMat
import time

class MeanFieldIsing():
    def __init__(self,ising_model):
        self.model = ising_model

    def run(self, maxIter=1000, eps = 0.000001, init_type='uniform'):
        if init_type == 'uniform':
            mean_val = np.zeros(self.model.num_var)
        elif init_type =='random':
            mean_val = 2 * (np.random.rand(self.model.num_var) - 0.5)

        for iter in range(maxIter):
            for i in range(self.model.num_var):
                tempVal = self.model.bias[i] + np.inner(self.model.coupling[:, i], mean_val)
                tempVal = (np.exp(2 * tempVal) - 1) / (np.exp(2 * tempVal) + 1)
                mean_val[i] = tempVal
        mean_val01 = (mean_val + 1) / 2
        ent = 0

        for i in range(self.model.num_var):
            if (mean_val01[i] > 1 - eps) | (mean_val01[i] < eps):
                ent = ent + 0
            else:
                ent = ent - mean_val01[i] * np.log(mean_val01[i]) - (1 - mean_val01[i]) * np.log(1 - mean_val01[i])

        inp = 0
        for i in range(self.model.num_var):
            inp = inp + self.model.bias[i] * mean_val[i]
            for j in range(i+1, self.model.num_var):
                inp = inp + self.model.coupling[i, j] * mean_val[i] * mean_val[j]

        log_z = inp + ent
        return log_z
