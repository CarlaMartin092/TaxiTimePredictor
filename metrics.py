import numpy as np

def accuracy(y_pred, y_true, interval = 2.0):
    acc = np.abs(y_true - y_pred)/60.0
    acc = acc <= interval
    return(np.mean(acc))