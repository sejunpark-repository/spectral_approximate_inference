import numpy as np

class IsingModel():
    def __init__(self,bias,coupling):
        self.bias = bias.copy()
        self.coupling = coupling.copy()
        self.num_var = coupling.shape[0]

