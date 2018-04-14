#! /usr/bin/env python
# A sample template for my_gaussian_system.py

import numpy as np
import scipy.io
import time
from my_gaussian_classify import *
from my_confusion import *

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1616497/data.mat";
data = scipy.io.loadmat(filename);

# Feature vectors: Convert uint8 to double, and divide by 255
Xtrn = data['dataset']['train'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
Xtst = data['dataset']['test'][0, 0]['images'][0, 0].astype(dtype=np.float_) / 255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0, 0]['labels'][0, 0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0, 0]['labels'][0, 0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

#YourCode - Prepare measuring time

# Run classification
epsilon = 0.01
#print 'Start: '
start = time.clock()
(Cpreds, Ms, Covs) = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)
#print 'Stop: '
print time.clock() - start
#YourCode - Measure the user time taken, and display it.

size_preds = np.shape(Cpreds)[0]
cm, a = my_confusion(Ctst, Cpreds)
#scipy.io.savemat('cm.mat', {'cm': cm})
#scipy.io.savemat('m26.mat', {'m26': Ms[:, 25]})
#scipy.io.savemat('cov26.mat', {'cov26': Covs[:, :, 25]})
print 'N    \t Nerrs\t Accuracy'
print size_preds, '\t', size_preds - np.sum(np.diagonal(cm)), '\t', a
#YourCode - Get a confusion matrix and accuracy

#YourCode - Save the confusion matrix as "Task3/cm.mat".

#YourCode - Save the mean vector and covariance matrix for class 26,
#           i.e. save Mu(:,25) and Cov(:,:,25) as "Task3/m26.mat" and
#           "Task3/cov26.mat", respectively.

#YourCode - Display the required information - N, Nerrs, acc.
