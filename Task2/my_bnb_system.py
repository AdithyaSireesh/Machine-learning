#! /usr/bin/env python
# A sample template for my_bnb_system.py

import numpy as np
import scipy.io
import time
from my_bnb_classify import *
from my_confusion import *
import matplotlib.pyplot as plt

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1616497/data.mat"
data = scipy.io.loadmat(filename)

# Feature vectors: Convert uint8 to double   (but do not divide by 255)
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float32)
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float32)
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

#YourCode - Prepare measuring time

# Run classification
threshold = 1.0

start = time.clock()
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
print 'Time taken: ', time.clock() - start
#
#plt.plot(thr, acc, 'ro', thr, acc, 'k')
#plt.ylabel('Accuracy')
#plt.xlabel('Threshold')
#plt.show()

size_preds = np.shape(Cpreds)[0]


#YourCode - Get a confusion matrix and accuracy
cm, a = my_confusion(Ctst, Cpreds)
#YourCode - Save the confusion matrix as "Task2/cm.mat".
#scipy.io.savemat('cm.mat', {'cm': cm})


#YourCode - Display the required information - N, Nerrs, acc.
print 'N    \t Nerrs\t Accuracy'
print size_preds, '\t', size_preds - np.sum(np.diagonal(cm)), '\t', a



