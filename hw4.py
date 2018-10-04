from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import graphviz


clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
#print(iris.data)
cross_val_score(clf, iris.data, iris.target, cv=5)