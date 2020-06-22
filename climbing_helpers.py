# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:27:16 2020

@author: nickp
"""

import sqlite3
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras import optimizers
from keras import models
from keras import layers

def scrape_database():
    conn = sqlite3.connect(r"C:\Users\nickp\Downloads\8a.sqlite")
    cursor = conn.cursor()
    
    # gets list of tuples, one tuble for each row in table
    # get climbers
    cursor.execute("SELECT * FROM user WHERE id < 2000")
    users = cursor.fetchall()
    
    # gets ascents 
    cursor.execute("SELECT * FROM ascent WHERE user_id < 2000")
    ascents = cursor.fetchall()
    
    # make ascents sorted by user_id => this is used when assigning avg user score
    ascents_sorted = sorted(ascents, key=lambda x: x[1])
    
    return ascents_sorted, users

def get_data(ascents_sorted, users):
    # scores are index 7 of each ascent row
    # user_id are index 1
    # loop through all ascents add make cumulative score for each user
    
    # initialize "empty" scores list
    user_scores = [0] * 2000
    
    # use this array to count number of climbs made by each user
    instances = [0] * 2000
    
    # safety table to keep track of user_ids
    user_ids = []
    
    # GET AVG_SCORE
    # step 1 - log cumulative scores and instances
    current_id = 0
    for ascent in ascents_sorted:
        
        if ascent[1] != current_id:
            user_ids.append(ascent[1])
            current_id = ascent[1]
            
        user_scores[ascent[1]] += ascent[7]
        instances[ascent[1]] += 1
        
    # step 2 - divide cumulative scores by number of climbs
    for i in range(len(instances)):
        
        if instances[i] != 0:
            user_scores[i] /= instances[i]
            
    # step 3 - filter out nonexistent users (where total score = 0)
    # CURRENTLY NOT NEEDED
    """
    user_scores_filtered = []
    
    for i in range(len(user_scores)):
        if i in user_ids:
            # TODO - figure out why this isn't a bell curve
            user_scores_filtered.append(user_scores[i])
    """
            
    # get labels for average score on flashes
    # need to get GRADES and SUCCESS data for each climb
    patterns = []
    targets = []
    
    for ascent in ascents_sorted:
        # pick metrics to analyze
        grade = ascent[2]
        method = ascent[5]
        user_id = ascent[1]
        
        # 0 for first try, 1 for multiple tries
        if method == 2 or method == 3 or method == 5:
            success = 0 
        else:
            success = 1
        
        patterns.append([user_scores[user_id], grade, instances[user_id]])
        targets.append(success)
        
    return patterns, targets

def shuffle(ordered_patterns, ordered_targets):
    # shuffle patterns and targets in the same way 
    
    patterns = np.asarray(ordered_patterns)
    targets = np.asarray(ordered_targets)
    
    indicies = np.arange(patterns.shape[0])
    np.random.shuffle(indicies)
    
    return patterns[indicies], targets[indicies]

def train_network(training_patterns, training_targets, n_hid):
    n_vox = training_patterns.shape[1]
    
    # multilayer network
    model = models.Sequential()
    model.add(layers.Dense(n_hid, activation="relu", input_shape=(n_vox,)))
    model.add(layers.Dense(n_hid, activation="relu"))
    model.add(layers.Dense(n_hid, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # use binary crossentropy loss since only two categories
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(training_patterns, training_targets, epochs=10)
    
    return model, history
    