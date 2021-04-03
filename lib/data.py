import numpy as np

class GenData:
    @staticmethod
    def _gen_linear(n=100):
        assert isinstance(n, (int, np.integer)), "Input should be 'Int'"
        assert n > 0, "Input should be 'positive'"

        inputs = np.random.uniform(0, 1, (n, 2))
        labels = []
        for inp in inputs:
            if inp[0] > inp[1]:
                labels.append(0)
            else:
                labels.append(1)
        
        return inputs, np.array(labels).reshape(-1, 1)

    @staticmethod
    def _gen_xor(n=100):
        assert isinstance(n, (int, np.integer)), "Input should be 'Int'"
        assert n > 0, "Input should be 'positive'"

        data_x = np.linspace(0, 1, n // 2)
        inputs, labels = [], []
        for x in data_x:
            inputs.append([x, x])
            labels.append(0)
            if x == 0.5:
                continue
            inputs.append([x, 1-x])
            labels.append(1)
        
        return np.array(inputs), np.array(labels).reshape(-1, 1)
    
def fetch_data(mode, n):
    assert mode == 'Linear' or mode == 'XOR'

    data_gen_func = {
        'Linear': GenData._gen_linear,
        'XOR': GenData._gen_xor
    }[mode]

    return data_gen_func(n)