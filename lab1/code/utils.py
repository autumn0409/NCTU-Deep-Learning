import numpy as np

def to_label(y_hat):
    y_hat_label = y_hat.copy()
    y_hat_label[y_hat_label > 0.5] = 1
    y_hat_label[y_hat_label < 0.5] = 0
    return y_hat_label

def cal_accuracy(y, y_hat):
    y_hat_label = to_label(y_hat)
    diff_mat = y - y_hat_label
    num_diff = np.count_nonzero(diff_mat)
    return 1.0 - (num_diff) / (y.shape[0])

