import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate
        
    def step(self, params, grad):
        for p, g in zip(params, grad):
            p -= g * self.lr
    
class SGD(Optimizer):
    pass
        
class Momentum(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.vel = None
    def step(self, params, grad):
        if isinstance(self.vel, type(None)):
            self.vel = np.array([np.zeros(params[0].shape), np.zeros(params[1].shape), np.zeros(params[2].shape)], dtype=object)
        for p, g, v in zip(params, grad, self.vel):
            v *= 0.9
            v -= g * self.lr
            p += v

class AdaGrad(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.lr_denom = None
        
    def step(self, params, grad):
        if isinstance(self.lr_denom, type(None)):
            self.lr_denom = np.array([
                np.zeros(params[0].shape), 
                np.zeros(params[1].shape), 
                np.zeros(params[2].shape)], dtype=object
            )
        for p, g, lrd in zip(params, grad, self.lr_denom):
            lrd += np.multiply(g, g)
            lr = self.lr / np.sqrt(lrd + 1e-4)
            p -= np.multiply(lr, g)
        
class Adam (Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.beta1, self.beta2 = 0.9, 0.999
        self.m_t, self.v_t, self.t = None, None, 0
        
    def step(self, params, grad):
        if isinstance(self.m_t, type(None)):
            self.m_t = np.array([np.zeros(params[0].shape), np.zeros(params[1].shape), np.zeros(params[2].shape)], dtype=object)
        if isinstance(self.v_t, type(None)):
            self.v_t = np.array([np.zeros(params[0].shape), np.zeros(params[1].shape), np.zeros(params[2].shape)], dtype=object)
            
        self.t += 1
        for g, m in zip(grad, self.m_t):
            m *= self.beta1
            m += (1 - self.beta1) * g
        for g, v in zip(grad, self.v_t):
            v *= self.beta2
            v += (1 - self.beta2) * np.multiply(g, g)
        
        m_hat = self.m_t / (1 - self.beta1 ** self.t)
        v_hat = self.v_t / (1 - self.beta2 ** self.t)
        
        for p, m, v in zip(params, m_hat, v_hat):
            p -= self.lr * np.divide(m, np.sqrt(v) + 1e-8)
        
class RMSProp (Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.denom = None
        self.t = 0
        
    def step(self, params, grad):
        self.t += 1
        if self.t == 1:
            self.denom = np.multiply(grad, grad)
        else:
            self.denom = 0.9 * self.denom + 0.1 * np.multiply(grad, grad)
        for p, g, d in zip(params, grad, self.denom):
            p -= self.lr * np.divide(g, np.sqrt(d) + 1e-8)

def fetch_optim(mode):
    assert mode == 'SGD' or mode == 'Momentum' or mode == 'AdaGrad' or mode == 'Adam' or mode == 'RMSProp'

    optimizer = {
        'SGD': SGD,
        'Momentum': Momentum,
        'AdaGrad': AdaGrad,
        'Adam': Adam,
        'RMSProp': RMSProp,
    }[mode]

    return optimizer