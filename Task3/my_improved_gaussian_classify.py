import numpy as np
from my_gaussian_classify import *

def my_improved_gaussian_classify(Xtrn, Ctrn, Xtst):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)
    # PERFORMING PCA
    trn_size, trn_feats = np.shape(Xtrn)
    test_size = np.shape(Xtst)[0]
    k = np.max(Ctrn) +1
    #M = np.zeros((k, trn_feats), dtype=np.float64)
    Cpreds = np.zeros((test_size, 1))
    mean_vector = np.sum(Xtrn, axis=0, keepdims=True, dtype=np.float32)*1.0/ trn_size
    covar = np.zeros((trn_feats, trn_feats))
    covar += (Xtrn - mean_vector).T.dot(Xtrn - mean_vector)

    #get eigen values and vectors
    eig_val_sc, eig_vec_sc = np.linalg.eig(covar)
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    K = 70
    matrix_w = np.zeros((trn_feats, K))
    for i in range(K):
        matrix_w[:, i] += eig_pairs[i][1]
    Xtrn = matrix_w.T.dot(Xtrn.T)
    Xtst = matrix_w.T.dot(Xtst.T)

    # applying gaussian classify to dimensionally reduced feature vectors
    Cpreds += my_gaussian_classify(Xtrn.T, Ctrn, Xtst.T, 0.01)[0]
    return Cpreds
