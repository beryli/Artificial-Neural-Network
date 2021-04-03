import numpy as np

class LossFunc:
    @staticmethod
    def _mse(y_gt, y_pred):
        return (y_pred - y_gt) ** 2
    
    @staticmethod
    def _der_mse(y_gt, y_pred):
        return y_pred - y_gt
    
    @staticmethod
    def _mae(y_gt, y_pred):
        return abs(y_pred - y_gt)
    
    @staticmethod
    def _der_mae(y_gt, y_pred):
        return np.where(y_pred > y_gt, 1, -1)
    
    @staticmethod
    def _ce(y_gt, y_pred):
        y_p, y_p_c = y_pred, 1 - y_pred
        if abs(y_p) < 1e-5:
            y_p = 1e-5
        if abs(y_p_c) < 1e-5:
            y_p_c = 1e-5
        return - y_gt * np.log(y_p) - (1 - y_gt) * np.log(y_p_c)
    
    @staticmethod
    def _der_ce(y_gt, y_pred):
        y_p, y_p_c = y_pred, 1 - y_pred
        if abs(y_p) < 1e-5:
            y_p = 1e-5 * np.where(y_p > 0, 1, -1)
        if abs(y_p_c) < 1e-5:
            y_p_c = 1e-5 * np.where(y_p_c > 0, 1, -1)
        return (- y_gt / (y_p)) + ((1 - y_gt) / (y_p_c))
    
def fetch_loss(mode):
    assert mode == 'mse' or mode == 'mae' or mode == 'ce'

    loss_func = {
        'mse': LossFunc._mse,
        'mae': LossFunc._mae,
        'ce': LossFunc._ce
    }[mode]

    der_loss_func = {
        'mse': LossFunc._der_mse,
        'mae': LossFunc._der_mae,
        'ce': LossFunc._der_ce
    }[mode]

    return loss_func, der_loss_func