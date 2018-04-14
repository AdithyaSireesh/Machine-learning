#! /usr/bin/env python
# A sample template for my_knn_system.py

import numpy as np
import scipy.io
from my_knn_classify import *
import my_knn_classify as knnc
from printTable import *

from my_confusion import *

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1616497/data.mat";
data = scipy.io.loadmat(filename)
import time
# Feature vectors: Convert uint8 to double, and divide by 255.
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float32) /255.0
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float32) /255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

#YourCode - Prepare measuring time

# Run K-NN classification
kb = [1, 3, 5, 10, 20]
size_kb = len(kb)

start = time.clock()
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)
size_preds = np.shape(Cpreds)[0]
#YourCode - Measure the user time taken, and display it.
print 'Time taken: ', (time.clock() - start)
#print (np.shape(Cpreds))
print '\n'
#YourCode - Get confusion matrix and accuracy for each k in kb.

#f0 = 'cm1.mat'
#f1 = 'cm3.mat'
#f2 = 'cm5.mat'
#f3 = 'cm10.mat'
#f4 = 'cm20.mat'
#fs = [f0, f1, f2, f3, f4]
dict = []
for i in range(size_kb):
    #YourCode - Save each confusion matrix.
    cm, a = my_confusion(Ctst, Cpreds[:, i].reshape((Ctst.size, 1)))
    #scipy.io.savemat(fs[i], {'cm': cm})
    dict.append({'k': kb[i], 'N': size_preds, 'Nerrs': size_preds - np.sum(np.diagonal(cm)), 'acc': a})

#YourCode - Display the required information - k, N, Nerrs, acc for each element of kb
printTable(dict)



