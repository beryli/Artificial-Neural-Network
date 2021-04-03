import numpy as np

class ActFunc:
    @staticmethod
    def _nan(x):
        return x
    
    @staticmethod
    def _der_nan(x):
        return 1
    
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _der_sigmoid( x):
        return np.multiply(ActFunc._sigmoid(x), 1.0 - ActFunc._sigmoid(x))
    
    @staticmethod
    def _ReLU(x):
        return np.where(x >= 0, x, x * 0.1)

    @staticmethod
    def _der_ReLU(x):
        return np.where(x >= 0, 1, 0.1)
    
    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _der_tanh(x):
        return 1 - np.tanh(x) ** 2
    
    
def fetch_act(mode):
    assert mode == 'sigmoid' or mode == 'nan' or mode == 'ReLU' or mode == 'tanh'

    act_func = {
        'nan': ActFunc._nan,
        'sigmoid': ActFunc._sigmoid,
        'ReLU': ActFunc._ReLU,
        'tanh': ActFunc._tanh
    }[mode]

    der_act_func = {
        'nan': ActFunc._der_nan,
        'sigmoid': ActFunc._der_sigmoid,
        'ReLU': ActFunc._der_ReLU,
        'tanh': ActFunc._der_tanh
    }[mode]

    return act_func, der_act_func