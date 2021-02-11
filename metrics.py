import numpy as np

def accuracy(y_pred, y_true, interval = 2.0):
    """
    Estimates the percentage of predicted values that are less than interval minutes away from their true value.

    Arguments:
    - y_pred: numpy.ndarray, values that are predicted by the model
    - y_true: numpy.ndarray, ground truth
    - interval: int, half of the size of the interval
    """
    acc = np.abs(y_true - y_pred)/60.0
    acc = acc <= interval
    return(np.mean(acc))