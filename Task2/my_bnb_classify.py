import numpy as np

def my_bnb_classify(Xtrn, Ctrn, Xtst, threshold):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   threshold   : A scalar threshold (type=float)
    # Output:
    #  Cpreds : N-by-1 ndarray of predicted labels for Xtst (dtype=np.int_)

    test_size = np.shape(Xtst)[0]
    trn_size, trn_feats = np.shape(Xtrn)

    #YourCode - binarisation of Xtrn and Xtst.
    Btrn = np.where(Xtrn < threshold, 0, 1)
    Btst = np.where(Xtst < threshold, 0, 1)

    k = np.max(Ctrn) +1
    #k = np.argmax(Ctrn.T)+1
    probs1 = np.zeros((k, trn_feats), dtype=np.float32)
    Cpreds = np.zeros((test_size, 1))
    pck = np.zeros((k, 1))
    for i in range(k):
        # choosing the training egs. belonging to class i
        bools = i == Ctrn.T[0]

        #grouping vectors of same class i
        ps = np.compress(bools, Btrn, axis=0)
        size_ps = np.shape(ps)[0]

        p = np.sum(ps, axis=0, keepdims=True, dtype=np.float32) / size_ps
        #pck[i] = size_ps*1.0/trn_size
        probs1[i] = p
    probs0 = 1-probs1

    #avoiding 0 probability problem
    probs1[probs1 == 0] = 1.0e-10
    probs0[probs0 == 0] = 1.0e-10
    probs1 = np.log(probs1)
    probs0 = np.log(probs0)

    #refer report for detailed explanation of this equation
    ci = np.dot(Btst, probs1.T) + np.dot((1-Btst), probs0.T)

    #get highest probabilities
    Cpreds = np.argmax(ci, axis=1).reshape((test_size, 1))

    #for i in range(test_size):
    #    ci = (probs1*Btst[i]) + probs0*(1-Btst[i])
    #    ci[ci == 0] = 1.0e-10
    #    ci = np.log(np.append(ci, pck, axis=1))
    #    #print 'log ci: ', ci
    #    ci = np.sum(ci, axis=1, keepdims=True, dtype=np.float)
    #    Cpreds[i] = np.argmax(ci)

    return Cpreds
