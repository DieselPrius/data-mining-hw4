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


tree_model = pickle.load(open('tree_model.txt', 'rb'))
knn_model = pickle.load(open('knn_model.txt', 'rb'))

testdf = pd.read_csv('irisData.csv')
data = testdf[["sepal_length","sepal_width","petal_length","petal_width"]].values

#print(testdf.iloc[[1]])

#pick 50 diffrent random indexes and create a test data set from them
randomIndexes = [] 
data_subset = [] #test data
for i in range(50):
    randomIndexes.append(random.randint(0,149))
    data_subset.append(data[randomIndexes[i]])


tree_predictions = tree_model.predict(data_subset)
knn_predictions = knn_model.predict(data_subset)


actualLabels = testdf["class"].values
subset_actualLabels = []
for i in range(len(randomIndexes)):
    subset_actualLabels.append(actualLabels[randomIndexes[i]])

#accuracy of tree
correctCount = 0
for i in range(len(tree_predictions)): #for each prediction
    if tree_predictions[i] == subset_actualLabels[i]:
        correctCount += 1

tree_acc = (correctCount / len(tree_predictions)) * 100 
#print("tree accuracy = " + str(tree_acc))

#acc of knn
correctCount = 0
for i in range(len(knn_predictions)): #for each prediction
    if knn_predictions[i] == subset_actualLabels[i]:
        correctCount += 1
knn_acc = (correctCount / len(knn_predictions)) * 100 
print("knn accuracy = " + str(knn_acc))

