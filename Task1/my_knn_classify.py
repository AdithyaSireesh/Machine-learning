import numpy as np
from scipy import stats
import time

def calculateDistance(train, test):
    #Euclidean distance calculation on numpy arrays
    #compute only the diagonal of the matrix product
    xx = np.einsum('ij,ji->i', test, test.T, dtype=np.float)
    yy = np.einsum('ij,ji->i', train, train.T, dtype=np.float)

    xx = np.reshape(xx, (np.shape(test)[0], 1))
    xy2 = -2*np.dot(test, train.T)

    DI = (xy2 + xx) + yy
    # Argsort sorts indices from closest to furthest neighbor, in ascending order
    sortDistance = np.argsort(DI, axis=1)
    #sortDistance += DI.argsort()
    return sortDistance

def test_classify_k(train, Ctrn, test, Ks):
    # Calculating nearest neighbours and based on majority vote assigning the class
    test_size = np.shape(test)[0]

    #get sorted matrices
    sortedDistIndices = calculateDistance(train, test)
    maxK = max(Ks)
    sortClasses = np.zeros((test_size, maxK))
    sortClasses += sortedDistIndices[:, :maxK]

    #lambda to convert indices to classes
    classer = lambda t: Ctrn[t]
    vfunc = np.vectorize(classer)
    sortClasses = vfunc(sortClasses)
    size_Ks = len(Ks)
    classes = np.zeros((test_size, size_Ks))

    # loop to get mode of the nearest neighbours for each k for each test feature vector
    for i in range(size_Ks):
        classes[:, i] = np.array(stats.mode(sortClasses[:, :Ks[i]], axis=1)[0]).T

    return classes

def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    # Calculating nearest neighbours and based on majority vote assigning the class
    Cpreds = test_classify_k(Xtrn, Ctrn, Xtst, Ks)
    return Cpreds