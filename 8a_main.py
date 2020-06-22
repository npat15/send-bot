# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:02:58 2020

@author: nickp
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras import optimizers
from keras import models
from keras import layers
import climbing_helpers as helpers

# get unfiltered data from SQLite database
ascents_sorted, users = helpers.scrape_database()

# get data and labels
ordered_patterns, ordered_targets = helpers.get_data(ascents_sorted, users)

# shuffle up data to ensure accurate cross-validation
patterns, targets = helpers.shuffle(ordered_patterns, ordered_targets)

# perform 5 folds
num_folds = 5

# make approximate partition size
stretch_index = patterns.shape[0] // num_folds

# create array for cross_validation indexing
runs = np.arange(num_folds)
runs = np.repeat(runs, stretch_index)

# add remaining runs to end of initial runs array
diff = patterns.shape[0] - runs.shape[0]

if diff != 0:
    added_inds = np.repeat(num_folds - 1, diff)
    runs = np.append(runs, added_inds)
    
# choose number of hidden nodes 
n_hid = 50

# store performances for each fold after training
performances = []

for i in range(num_folds):
    test_run = i
    
    print("FOLD", test_run + 1)
    
    # select training and testing patterns/labels
    train_these = runs != test_run
    test_these = runs == test_run
    
    training_patterns = patterns[train_these]
    training_targets = targets[train_these]
    
    testing_patterns = patterns[test_these]
    testing_targets = targets[test_these]
    
    # train network and get performance
    model, history = helpers.train_network(training_patterns, training_targets, n_hid)
    eval_loss, eval_perf = model.evaluate(testing_patterns, testing_targets)
    
    # record performance
    performances.append(eval_perf)
    
print("Performances:", performances)
print("Average Performance:", np.average(performances))

