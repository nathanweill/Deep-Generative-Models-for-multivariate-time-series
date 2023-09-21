import numpy as np
import pandas as pd
from math import*
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import glob
import sys

def to_padded(arr, zero=False, pred=True):
    """Assumes a 3-dimensional array ; returns the array of sequences at the right time position, padded with nans"""
    sh1, sh22, sh3 = arr.shape
    if pred:
        sh2 = sh22
    else:
        sh2 = sh22 + 1

    if zero:
        res = np.zeros((sh1, sh1+sh2, sh3))
    else:
        res = np.full((sh1, sh1+sh2, sh3), np.nan)
        
    for k in range(sh1):
        res[k, k:k+sh2, :] = arr[k]
    
    return res

def from_padded(arr, pred=True):
    """Assumes a 3-dimensional array ; returns the array of sequences depadded"""
    sh1, sh22, sh3 = arr.shape
    if pred:
        sh2 = sh22 - sh1
    else: 
        sh2 = sh22 -sh1 - 1
    res = np.full((sh1, sh2, sh3), np.nan)
    for k in range(sh1):
        res[k, :, :] = arr[k, k:k+sh2, :]
    
    return res


def rebuild(obs_data, updates_gen):
    """
    Yields forecast trajectories from a given observation dataset and generated prediction updates
    args:
    obs_data: array of shape (n_samples, dimension), e.g. full year of observations 1h sampled
    updates_gen: array of shape (n_samples, seq_size, dimension), e.g. series of updates for 46h datapoints per sample
    return: array of shape (n_samples-seq_size, seq_size, dimension)
    """
    n, m, d = updates_gen.shape
    n += 1
    print(n, m, d)
    up_padded =  to_padded(updates_gen, zero=True)
    traj_rebuilt = np.zeros((n-m, n, d))

    for i in range(n-m):
        for j in range(i, i+m):
            traj_rebuilt[i, j] = obs_data[j] - sum([up_padded[k, j] for k in range(i, j)])

    res = from_padded(traj_rebuilt)
    return res

