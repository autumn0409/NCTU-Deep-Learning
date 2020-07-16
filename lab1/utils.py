import numpy as np

def to_label(y_pred):
    y_pred_label = y_pred.copy()
    y_pred_label[y_pred_label > 0.5] = 1
    y_pred_label[y_pred_label < 0.5] = 0
    return y_pred_label

def cal_accuracy(y, y_pred):
    y_pred_label = to_label(y_pred)
    diff_mat = y - y_pred_label
    num_diff = np.count_nonzero(diff_mat)
    return 1.0 - (num_diff) / (y.shape[0])

