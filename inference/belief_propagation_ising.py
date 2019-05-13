import numpy as np
from gm.ising_model import IsingModel
from read_mat import ReadMat
import time

class BeliefPropagationIsing():
    def __init__(self,ising_model):
        self.model = ising_model
        self.filter00, self.filter01 = np.nonzero(self.model.coupling)
        self.filter10, self.filter11 = np.where(self.model.coupling == 0)

    def run(self, maxIter=1000, damping=0.0, init_type='uniform'):
        h2 = np.tile(self.model.bias, (self.model.num_var, 1))
        A = self.model.coupling
        n = self.model.num_var
        b_old = np.zeros((self.model.num_var, 2))
        bb_old = np.zeros((self.model.num_var, self.model.num_var, 4))
        if init_type == 'uniform':
            m1_old = np.zeros((self.model.num_var,self.model.num_var))
        elif init_type == 'random':
            m1_old = np.random.randn(self.model.num_var, self.model.num_var)

        for iter in range(maxIter):
            m1 = self.update_message(m1_old, h2, damping)

            m1_old = m1

        b, bb = self.calculate_bp_marginal(m1, h2)
        log_z = self.get_log_z(b, bb)
        return log_z

    def get_log_z(self, b, bb):
        hh = np.zeros((self.model.num_var, 2))
        hh[:, 0] = -1 * self.model.bias
        hh[:, 1] = self.model.bias
        Ub = -1 * b * hh

        AA = np.zeros((self.model.num_var, self.model.num_var, 4))
        AA[:, :, 0] = self.model.coupling
        AA[:, :, 1] = -1 * self.model.coupling
        AA[:, :, 2] = -1 * self.model.coupling
        AA[:, :, 3] = self.model.coupling
        Ubb = -1 * bb * AA

        U = np.sum(Ub) + np.sum(Ubb) / 2

        bbb = np.zeros((self.model.num_var, self.model.num_var, 4))

        bbb[:, :, 0] = bb[:, :, 0] + bb[:, :, 1]
        bbb[:, :, 1] = bb[:, :, 2] + bb[:, :, 3]
        bbb[:, :, 2] = bb[:, :, 0] + bb[:, :, 2]
        bbb[:, :, 3] = bb[:, :, 1] + bb[:, :, 3]

        for i in range(self.model.num_var):
            if b[i, 0] == 0:
                b[i, 0] = 1

            if b[i, 1] == 0:
                b[i, 1] = 1

        for i in range(self.model.num_var):
            for j in range(self.model.num_var):
                for k in range(4):
                    if bb[i, j, k] == 0:
                        bb[i, j, k] = 1

                    if bbb[i, j, k] == 0:
                        bbb[i, j, k] = 1

        Hb = -1 * b * np.log(b)
        Hbb = -1 * bb * np.log(bb)
        Hbbb = -1 * bbb * np.log(bbb)

        Ibb = np.sum(Hbbb) - np.sum(Hbb)

        ent = -1 * Ibb / 2 + np.sum(Hb)

        log_z = ent - U
        return log_z

    def update_message(self, m1_old, h2, damping=0.0):
        temp1 = np.sum(m1_old, axis=0)

        temp1 = np.tile(temp1, (self.model.num_var, 1)) - m1_old - h2
        temp2 = h2

        temp11 = temp1 + self.model.coupling
        temp12 = temp1 - self.model.coupling
        temp21 = temp2 - self.model.coupling
        temp22 = temp2 + self.model.coupling

        m1 = self.add_large_logs(temp11, temp21)
        m2 = self.add_large_logs(temp12, temp22)

        m1 = np.transpose(m1)
        m2 = np.transpose(m2)
        m1 = m1 - m2
        m1 = damping * m1_old + (1 - damping) * m1

        m1 = self.clear_zeros(m1)

        return m1

    def calculate_bp_marginal(self, m, h2):
        b = np.zeros((self.model.num_var, 2))
        bb = np.zeros((self.model.num_var, self.model.num_var, 4))

        temp1 = np.sum(m, axis=0)

        b[:,0] = temp1 - self.model.bias
        b[:,1] = self.model.bias

        temp1 = np.sum(m, axis=0)
        temp1 = np.tile(temp1, (self.model.num_var, 1)) - m - h2
        temp2 = h2

        temp11 = temp1 + np.transpose(temp1) + self.model.coupling
        temp12 = temp2 + np.transpose(temp1) - self.model.coupling
        temp21 = temp1 + np.transpose(temp2) - self.model.coupling
        temp22 = temp2 + np.transpose(temp2) + self.model.coupling

        temp12 = np.tril(temp12, k=-1) + np.transpose(np.tril(temp12, k=-1))
        temp21 = np.tril(temp21, k=-1) + np.transpose(np.tril(temp21, k=-1))

        bb[:, :, 0] = temp11
        bb[:, :, 1] = temp12
        bb[:, :, 2] = temp21
        bb[:, :, 3] = temp22

        b, bb = self.normalize_marginal(b, bb)

        return b, bb

    def normalize_marginal(self, b, bb):
        for i in range(self.model.num_var):
            if np.abs(b[i, 0] - b[i, 1]) > 100:
                if b[i, 0] > b[i, 1]:
                    b[i, 0] = 1
                    b[i, 1] = 0
                else:
                    b[i, 0] = 0
                    b[i, 1] = 1

            else:
                mn = np.min(b[i, :])
                b[i, 0] = np.exp(b[i, 0] - mn)
                b[i, 1] = np.exp(b[i, 1] - mn)

        for i in range(self.filter00.shape[0]):
            idx0 = self.filter00[i]
            idx1 = self.filter01[i]

            mx = np.max(bb[idx0, idx1, :])

            for k in range(4):
                if mx - bb[idx0, idx1, k] > 100:
                    bb[idx0, idx1, k] = 0
                else:
                    bb[idx0, idx1, k] = np.exp(bb[idx0, idx1, k] - mx + 100)

        bb[:, :, 0] = np.tril(bb[:, :, 0]) + np.transpose(np.tril(bb[:, :, 0])) + np.identity(self.model.num_var)
        bb[:, :, 1] = np.tril(bb[:, :, 1]) + np.transpose(np.tril(bb[:, :, 1])) + np.identity(self.model.num_var)
        bb[:, :, 2] = np.tril(bb[:, :, 2]) + np.transpose(np.tril(bb[:, :, 2])) + np.identity(self.model.num_var)
        bb[:, :, 3] = np.tril(bb[:, :, 3]) + np.transpose(np.tril(bb[:, :, 3])) + np.identity(self.model.num_var)

        sum_b = np.sum(b, axis=1)
        sum_bb = np.sum(bb, axis=2)
        sum_b = np.reshape(sum_b, (self.model.num_var, 1))
        sum_b = np.tile(sum_b, (1, 2))
        sum_bb = np.repeat(sum_bb[:, :, np.newaxis], 4, axis=2)

        b = b / sum_b
        bb = bb / sum_bb

        for i in range(self.filter10.shape[0]):
            idx0 = self.filter10[i]
            idx1 = self.filter11[i]

            bb[idx0, idx1, 0] = 0
            bb[idx0, idx1, 1] = 0
            bb[idx0, idx1, 2] = 0
            bb[idx0, idx1, 3] = 0

        return b, bb

    def add_large_logs(self, A, B, threshold=100):
        C = np.zeros((self.model.num_var, self.model.num_var))

        for i in range(self.filter00.shape[0]):
            idx0 = self.filter00[i]
            idx1 = self.filter01[i]

            a = A[idx0, idx1]
            b = B[idx0, idx1]

            if a < b:
                temp_val = a
                a = b
                b = temp_val

            if a - b > threshold:
                C[idx0, idx1] = a
            else:
                C[idx0, idx1] = b + np.log(1 + np.exp(a - b))

        return C



    def clear_zeros(self, m):
        for i in range(self.filter10.shape[0]):
            m[self.filter10[i], self.filter11[i]] = 0

        return m