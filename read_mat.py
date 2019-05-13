import scipy.io as sio
import numpy as np
from gm.gm import GraphicalModel
from gm.ising_model import IsingModel
from gm.factor import Factor
from copy import copy
# from inference.bucket_elimination import BucketElimination
# from inference.mini_bucket_elimination import MiniBucketElimination
# from inference.weighted_mini_bucket_elimination import WeightedMiniBucketElimination
# from graphical_model.models.ising_grid import ising_grid
# from inference.belief_propagation import BeliefPropagation



def factor_name(i,j=-1):
    if j == -1:
        return 'B' + str(i)
    else:
        return 'F' + str(i) + str(j)

def variable_name(i):
    return 'V' + str(i)

class ReadMat():
    def __init__(self,filename):
        super(ReadMat, self).__init__()


        self.mat_contents = sio.loadmat(filename)
        self.coupling = self.mat_contents['A']
        self.logZ = self.mat_contents['logZ']
        self.bias = self.mat_contents['h']
        self.nv = self.coupling.shape[1]

        if self.coupling.shape[0] != self.coupling.shape[1]:
            print('ReadMat: input is not square', self.coupling.shape[0], self.coupling.shape[1])

    def mat2gm(self, cw,sample):
        fg = GraphicalModel()
        if len(self.coupling.shape) > 3:
            A = self.coupling[:,:,cw,sample]
            h = self.bias[:,cw,sample]
        else:
            if cw == 0 & sample == 0:
                A = self.coupling
                h = self.bias
            else:
                print('ReadMat: mat2gm: invalid cw, sample')

        for i in range(self.nv):
            fg.add_variable(variable_name(i))
            factor = Factor(name = factor_name(i),
                            variables = [variable_name(i)],
                            log_values = np.array([-h[i], h[i]]))
            fg.add_factor(factor)

        for i in range(self.nv):
            for j in range(i):
                if A[i,j] != 0:
                    v1 = variable_name(i)
                    v2 = variable_name(j)
                    beta = A[i,j]
                    factor = Factor(name = factor_name(i,j),
                                    variables = [v1, v2],
                                    log_values = np.matrix([[beta, -beta], [-beta, beta]]))
                    fg.add_factor(factor)

        return fg

    def mat2im(self,cw,sample):
        return IsingModel(self.bias[:,cw,sample],self.coupling[:,:,cw,sample])

    def get_sdp_eig(self,cw,sample):
        return self.mat_contents['Ux'][:,:,cw,sample], np.diag(self.mat_contents['Sx'][:,:,cw,sample])

    def get_sdp_time(self,cw,sample):
        return self.mat_contents['time_sdp'][cw,sample]

    def get_cwrange(self):
        return self.mat_contents['cw_range']

    def get_num_sample(self):
        return self.mat_contents['nSample'][0][0]
