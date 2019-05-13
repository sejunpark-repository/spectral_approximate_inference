import numpy as np
from gm.ising_model import IsingModel

class Rank1Ising():
    def __init__(self,ising_model,V,s):
        self.model = ising_model
        self.s = s.copy()
        self.V = V.copy()

    def run_rank1(self, q=0.001, use_gaus=False):
        if use_gaus:
            log_z = self.model.num_var * np.log(2) - 0.5 * np.log(1-self.s).sum() - 0.5 * sum(self.s)
        else:
            log_zz = 0
            for i in range(self.model.num_var+1):
                temp = self.rank_1_quantize(self.V[:,i], self.s[i], q=q)
                log_zz += temp
            log_z = log_zz - (self.model.num_var+1) * self.model.num_var * np.log(2) - np.log(2) - 0.5 * sum(self.s)
        return log_z

    def run_rankr(self, threshold=50, num_sample=10000, use_gaus=False):
        log_z = 0
        s2 = -1 * self.s
        idx0 = 0
        idx1 = 0
        val = 0
        for i in range(self.model.num_var+1):
            if val + s2[i] > threshold:
                log_z += self.get_local_log_z(idx0, idx1, s2, num_sample, use_gaus) - (self.model.num_var+1) * np.log(2)
                idx0 = idx1
                val = 0
            val += s2[i]
            idx1 += 1

            if i == self.model.num_var:
                log_z += self.get_local_log_z(idx0, idx1, s2, num_sample, use_gaus) - (self.model.num_var + 1) * np.log(2)

        return self.model.num_var * np.log(2) + log_z - 0.5 * sum(self.s)

    def get_local_log_z(self, idx0, idx1, s2, num_sample, use_gaus):
        if idx1 - idx0 > 1:
            return self.rank_r_rand(self.V[:, idx0:idx1],
                                      self.s[idx0:idx1],
                                      num_sample)
        elif use_gaus:
            return self.rank_1_gaus(self.s[idx0])

        else:
            return self.rank_1_quantize(self.V[:, idx0],
                                        self.s[idx0])

    def rank_1_gaus(self, s):
        log_z = (self.model.num_var+1) * np.log(2) - 0.5 * np.log(1-s)
        return log_z

    def rank_1_quantize(self, v, s, q=0.001):
        n = v.shape[0]
        max_val = v[v>0].sum() + n * q
        min_val = v[v<0].sum() - n * q
        len = 2 * int(round((max_val - min_val) / q)) + 1

        idxv = (2 * v / q).round().astype(int)

        memory = np.zeros(len)
        idx0 = int(round((len  - 1) / 2)) + int(round(-1 * v.sum() / q))
        memory[idx0] = 1

        # t1 = time.time()

        for i in range(n):
            start_idx1, end_idx1 = self.get_range(len, idxv[i])
            start_idx2, end_idx2 = self.get_range(len, -1 * idxv[i])
            memory[start_idx1:end_idx1] += memory[start_idx2:end_idx2]

        idx_nonzero = np.nonzero(memory)

        memory2 = memory[idx_nonzero[0]]
        val = min_val - max_val + idx_nonzero[0] * q
        temp = memory2 * np.exp(s / 2 * np.square(val))
        log_z = np.log(temp.sum())

        return log_z

    def rank_1_quantize2(self, v, s, h, q=0.001):
        n = v.shape[0]
        max_val = v[v>0].sum() + n * q
        min_val = v[v<0].sum() - n * q
        len = 2 * int(round((max_val - min_val) / q)) + 1

        idxv = (2 * v / q).round().astype(int)

        memory = np.zeros(len)
        idx0 = int(round((len  - 1) / 2)) + int(round(-1 * v.sum() / q))
        memory[idx0] = 1


        for i in range(n):
            start_idx1, end_idx1 = self.get_range(len, idxv[i])
            start_idx2, end_idx2 = self.get_range(len, -1 * idxv[i])
            memory[start_idx1:end_idx1] = exp(h[i]) * memory[start_idx2:end_idx2] + exp(-h[i]) * memory[start_idx1:end_idx1]

        idx_nonzero = np.nonzero(memory)

        memory2 = memory[idx_nonzero[0]]
        val = min_val - max_val + idx_nonzero[0] * q
        temp = memory2 * np.exp(s / 2 * np.square(val))
        log_z = np.log(temp.sum())

        return log_z

    def get_range(self, len, idx):
        start_idx = np.amax([0, idx])
        end_idx = np.amin([len, len + idx])
        return start_idx, end_idx

    def add_large_logs(self, a, b, threshold=300):
        if b > a:
            temp = a
            a = b
            b = temp
        if a - b > threshold:
            return a
        else:
            return a + np.log(1 + np.exp(b - a))


    def rank_r_rand(self, V, s, num_sample=10000):
        if V.shape[1] != s.shape[0]:
            print('Rank1Ising: rank_r_rand: Size of V, s does not match')
            return
        else:
            r = V.shape[1]
            U = np.matmul(V,np.diag(np.sqrt(-1*s)))
            sample = np.random.randn(r,num_sample)
            temp_val = np.cos(np.matmul(U,sample))
            temp_z = np.prod(temp_val, axis=0)
            z1 = np.mean(temp_z)
            z2 = np.median(temp_z)

            if z1 > 0:
                log_z = (self.model.num_var + 1) * np.log(2) + np.log(z1)
            elif z2 > 0:
                log_z = (self.model.num_var + 1) * np.log(2) + np.log(z2)
            else:
                print('Rank1Ising: both mean and median are negative')
                log_z = (self.model.num_var + 1) * np.log(2) + np.log(np.abs(z1))
            return log_z
