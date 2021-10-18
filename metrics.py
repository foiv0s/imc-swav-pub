from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment as linear_assignment

ari = adjusted_rand_score
nmi = normalized_mutual_info_score


def acc(y_true, y_pred, detailed=False):
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    if detailed:
        return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size, w, ind
    else:
        return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size
