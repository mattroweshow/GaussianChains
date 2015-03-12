__author__ = 'rowem'

import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_mldata
from gaussian_chain import SingleGaussianChain, DoubleGaussianChain
from sklearn.naive_bayes import MultinomialNB

# From remote server
custom_data_home = ""
data = fetch_mldata('covtype.binary_scale', data_home=custom_data_home)

# Use only the first 1000 instances for now
max_instance_number = 1000
instance_ids = range(0, max_instance_number)

# Only use the first 3 features for now
# data_X = data.data[[instance_ids],[feature_ids]]
data_X = data.data[instance_ids,:]

data_y = data.target[instance_ids]
# Relabel the y values to be 0 or 1
data_y_b = np.mod(1, data_y)
data_y = data_y_b

# Specify the numeric target class
target_class = 1

# Prepare the data for experiments
np.random.seed(0)
indices = np.random.permutation(data_X.shape[0])

# Get the 90% cutoff in terms of absolute indices
cutoff = int(data_X.shape[0] * 0.1)
# Randomly split into training and testing
X_train = data_X[indices[:-cutoff]]
y_train = data_y[indices[:-cutoff]]
X_test = data_X[indices[-cutoff:]]
y_test = data_y[indices[-cutoff:]]

# Set the hyperparameters for the Gaussian Chain Models
eta = 0.001
lambdA = 0.001
rho = 0.9
alpha = 0.5

# Set the learning mode: 1 = single stochastic gradient descent, 2 = double stochastic gradient descent
learning_mode = 2

# Baseline Model 1
print "Naive Bayes"
clf = MultinomialNB().fit(X_train, y_train)
predicted_nb = clf.predict(X_test)
print(metrics.roc_auc_score(y_test, predicted_nb))

# Model 1
print "Single Gaussian Chain"
gcm = SingleGaussianChain(rho, alpha, lambdA, eta, learning_mode, target_class)
gcm.fit(X_train, y_train)
predicted_gcm = gcm.predict_proba(X_test)
print(metrics.roc_auc_score(y_test, predicted_gcm))

# Model 2
print "Double Gaussian Chain"
gcm = DoubleGaussianChain(rho, alpha, lambdA, eta, learning_mode, target_class)
gcm.fit(X_train, y_train)
predicted_gcm = gcm.predict_proba(X_test)
print(metrics.roc_auc_score(y_test, predicted_gcm))



