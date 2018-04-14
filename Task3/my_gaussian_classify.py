import numpy as np
import logdet as ld

def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   epsilon   : A scalar parameter for regularisation (type=float)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=int_)
    #  Ms    : D-by-K ndarray of mean vectors (dtype=np.float_)
    #  Covs  : D-by-D-by-K ndarray of covariance matrices (dtype=np.float_)

    trn_size, trn_feats = np.shape(Xtrn)
    test_size = np.shape(Xtst)[0]
    k = 26
    #k = 2
    Ms = np.zeros((k, trn_feats), dtype=np.float64)

    Cpreds = np.zeros((test_size, 1))

    Covs = np.zeros((trn_feats, trn_feats, k))

    for i in range(k):
        # choosing the training egs. belonging to class i
        bools = i == Ctrn.T[0]
        ps = np.compress(bools, Xtrn, axis=0)
        Ms[i] = np.sum(ps, axis=0, dtype=np.float64)/ np.shape(ps)[0]

        #compute covariance of class i
        Covs[:, :, i] = np.dot((ps - Ms[i]).T, ps - Ms[i])*1.0 / np.shape(ps)[0]
        Covs[:, :, i] += np.identity(trn_feats)*epsilon

    lll = np.zeros((test_size, k))
    for i in range(k):
        X = np.dot(Xtst - Ms[i], np.linalg.inv(Covs[:, :, i]))
        Y = (Xtst - Ms[i]).T

        #computing log posterior probabilities
        lll[:, i] += -0.5*(ld.logdet(Covs[:, :, i])) - 0.5*np.einsum('ij,ji->i', X, Y)
    for i in range(test_size):
        # picking class with highest probability as prediction
        Cpreds[i][0] = lll[i].argsort()[-1]
    Ms = Ms.T

    return (Cpreds, Ms, Covs)

