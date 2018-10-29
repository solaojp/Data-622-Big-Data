#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:02:01 2018

@author: GD
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import re
from matplotlib import pyplot as plt
from matplotlib import style



# Load Dataset

traindata = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")


# Checks on statistics of data given
traindata.info()
traindata.describe()

#The training-set has 891 examples and 11 features + the target variable (survived). 
#2 of the features are floats, 5 are integers and 5 are objects

# Data preprocessing

table1 = traindata.groupby(traindata['Survived']).count()['PassengerId']
table2 = traindata.groupby(traindata['Age'].isnull()).mean()['Survived']

#Recognising missing data
#we took care of it by aggreagating values while building the model.

total = traindata.isnull().sum().sort_values(ascending=False)
percent_1 = traindata.isnull().sum()/traindata.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])


# Plots 

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = traindata[traindata['Sex']=='female']
men = traindata[traindata['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# Class-age-gender model
# Age fare model

#As we are heading towards our model creation,we will check the effect of class on survival

sns.barplot(x='Pclass', y='Survived', data=traindata)

# Dealing with missing data

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [traindata, testdata]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
traindata = traindata.drop(['Cabin'], axis=1)
testdata = testdata.drop(['Cabin'], axis=1)
