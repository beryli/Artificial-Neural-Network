import numpy as np
from .activation import fetch_act
from .loss import fetch_loss
from .optimizer import fetch_optim

class ANN:
    def __init__(self, hidden_size=5, learning_rate=1, batch_size=1, act_func="sigmoid", loss_func="mse", optimizer="SGD", num_step=2000, print_interval=100):
        self.lr = learning_rate
        self.bs = batch_size
        
        self.act,  self.der_act = fetch_act(act_func)
        self.loss,  self.der_loss = fetch_loss(loss_func)
        self.optim = fetch_optim(optimizer)(self.lr)
        
        self.num_step = num_step
        self.print_interval = print_interval

        self.hidden1_weights = np.random.normal(0, 0.5, (2, hidden_size))
        self.hidden2_weights = np.random.normal(0, 0.5, (hidden_size, hidden_size))
        self.hidden3_weights = np.random.normal(0, 0.5, (hidden_size, 1))

    def forward(self, inputs):
        self.h1 = np.dot(inputs, self.hidden1_weights)
        self.z1 = self.act(self.h1)
        self.h2 = np.dot(self.z1, self.hidden2_weights)
        self.z2 = self.act(self.h2)
        self.h3 = np.dot(self.z2, self.hidden3_weights)
        self.output = self.act(self.h3)

        return self.output
    
    def backward(self, inputs):
        ''' dimension
        input, output: [bs, 2], [bs, 1]
        g3, g2, g1 = [bs, 1], [bs, hidden_num], [bs, hidden_num]
        z2, z1, inputs = [bs, hidden_num], [bs, hidden_num], [1, 2]
        d3, d2, d1 = [hidden_num, 1], [hidden_num, hidden_num], [2, hidden_num]
        '''
        g3 = np.multiply(self.error, self.der_act(self.h3))
        g2 = np.multiply(np.dot(g3, self.hidden3_weights.T), self.der_act(self.h2))
        g1 = np.multiply(np.dot(g2, self.hidden2_weights.T), self.der_act(self.h1))
        
        grad3 = np.dot(self.z2.T, g3) / self.curr_bs # from hidden layer to output layer
        grad2 = np.dot(self.z1.T, g2) / self.curr_bs
        grad1 = np.dot(inputs.T, g1) / self.curr_bs # from input layer to hidden layer

        self.optim.step(
            np.array([
                self.hidden1_weights, 
                self.hidden2_weights,
                self.hidden3_weights
            ], dtype=object), 
            np.array([grad1, grad2, grad3], dtype=object)
        )

        
    def train(self, inputs, labels):
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]
        loss_list = []
        for epochs in range(1, self.num_step+1):
            for idx in range(0, n, self.bs):
                self.curr_bs =  n - idx if idx + self.bs >= n else self.bs
                self.output = self.forward(inputs[idx:idx+self.curr_bs, :])
                self.error = self.der_loss(labels[idx:idx+self.curr_bs, :], self.output)
                self.backward(inputs[idx:idx+self.curr_bs, :])

            loss_sum = 0.0
            for idx in range(inputs.shape[0]):
                output = self.forward(inputs[idx:idx+1, :])
                loss_sum += self.loss(labels[idx:idx+1, :], output)
            loss_list.append(loss_sum/n)
            
            if epochs % self.print_interval == 0 or epochs == 1:
                print('Epochs %4d, ' % epochs + 'loss %8.5f' % (loss_sum/n))

        print('Training finished')
        return np.array(loss_list).reshape(-1)
        
        
    def test(self, inputs, labels):
        assert inputs.shape[0] == labels.shape[0]
        
        pred_y = np.round(self.forward(inputs))
        n = labels.shape[0]
        err = 0
        for i, j in zip(labels, pred_y):
            if i != j:
                err += 1
        acc = ((1 - (err / n)) * 100)
        print("Accuracy: %.2f" % acc + '%')
        return pred_y, ("%.2f" % acc + '%')