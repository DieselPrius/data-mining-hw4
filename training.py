from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import graphviz
import pickle
from sklearn.neighbors import KNeighborsClassifier
import random

#read in data and seperate it as needed
traindf = pd.read_csv('irisData.csv')
data = traindf[["sepal_length","sepal_width","petal_length","petal_width"]].values
labels = traindf["class"].values

randomIndexes = [] 
data_subset = [] 
labels_subset = []
for i in range(50):
    randomIndexes.append(random.randint(0,149))
    data_subset.append(data[randomIndexes[i]])
    labels_subset.append(labels[randomIndexes[i]])


#create and save tree
estimator = DecisionTreeClassifier()
estimator.fit(data_subset,labels_subset)
filename = 'tree_model.txt'
pickle.dump(estimator, open(filename,'wb'))

#create and save knn estimator
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(data_subset, labels_subset) 
filename = 'knn_model.txt'
pickle.dump(estimator, open(filename,'wb'))


