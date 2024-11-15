import csv
import pandas as pd
import sys
import math
import numpy as np
import string
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    # takes in a string with a path to a CSV file 
    #and returns the
    #data points as a list of dicts. Section 0.1
    
    lodaded_data = []
    with open(filepath, mode='r', encoding = 'utf-8') as file:
        r = csv.DictReader(file)
        for row in r :
            loaded_data.append(dict(row))
    return loaded_data



calc_features(row) :
    #takes in one row dict from the data loaded from the previous
# function then calculates the corresponding feature vector for that country as
# specified above, and returns it as a NumPy array of shape (6,). The dtype of this
# array should be float64. Section 0.2


hac(features) :
    # performs single linkage hierarchical agglomerative clustering on
# the country with the (x1
# , . . . , x6
# ) feature representation, and returns a NumPy
# array representing the clustering. Section 0.3


fig_hac(Z, names) :
    #visualizes the hierarchical agglomerative clustering on the
# country’s feature representation. Section 0.4


normalize_features(features) : 
#     — takes a list of feature vectors and computes the
# normalized values. The output should be a list of normalized feature vectors in the
# same format as the input. Section 0.5


# +
#q1) Model : ChatGPT, prompt: how to load data through a filepath with dictreader
#q2) Model : ChatGPT, prompt: 

