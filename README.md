# GaussianChains

This repository contains the Gaussian Chains models used for binary classification. The models function by assuming that each feature in the training dataset has a Gaussian distribution, and then learns the contribution of each Gaussian to classification outcome.

The co-domain of each model is a probability value constrained to the closed interval [0,1], hence the models do not produce classification labels - however these can be inferred through a switch function.

This code has been published in the following paper:
- [Predicting Online Community Churners using Gaussian Sequences](http://www.lancaster.ac.uk/staff/rowem/files/mrowe-socinfo2014.pdf). M Rowe. In the proceedings of the 6th International Conference on Social Informatics. Barcelona, Spain. (2014)

The code is contained within the gaussin_chain.py file where there are two models for usage:

1. Single Gaussian Chains - where the Gaussian distributions of the target class's training instances' features are only considered.

2. Double Gaussian Chains - where the target class, and non-target class, Gaussian distributions are considered.

Each model offers two learning procedures:

1. Single Stochastic Gradient Descent - where learning is performed by shuffling the order of the training instances each learning epoch.

2. Dual Stochastic Gradient Descent - where both the order of the training instances and the feature indices are shuffled eahc learning epoch.