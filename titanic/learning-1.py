#! /usr/bin/env python3

"""
https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""

from shared.utils import *

# data handling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train = pd.read_csv("./input/train.csv") # size: 891
test = pd.read_csv("./input/test.csv") # size: 418

print(train.info())

# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64
# SibSp          891 non-null int64 # ?
# Parch          891 non-null int64 # ?
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object # ?
# Embarked       889 non-null object # ?
# dtypes: float64(2), int64(5), object(5)

print(train.describe())

#         PassengerId   Survived    Pclass      Age         SibSp      Parch        Fare
# count   891.000000    891.000000  891.000000  714.000000  891.000000 891.000000  891.000000
# mean    446.000000      0.383838    2.308642   29.699118    0.523008   0.381594   32.204208
# std     257.353842      0.486592    0.836071   14.526497    1.102743   0.806057   49.693429
# min       1.000000      0.000000    1.000000    0.420000    0.000000   0.000000    0.000000
# 25%     223.500000      0.000000    2.000000   20.125000    0.000000   0.000000    7.910400
# 50%     446.000000      0.000000    3.000000   28.000000    0.000000   0.000000   14.454200
# 75%     668.500000      1.000000    3.000000   38.000000    1.000000   0.000000   31.000000
# max     891.000000      1.000000    3.000000   80.000000    8.000000   6.000000  512.329200
