#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:02:32 2018

@author: GD
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


#class-age-gender model

traindata = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")
x = traindata.iloc[:,[2,4,5]].values
y = traindata.iloc[:,1].values

# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0])


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 0:3])
x[:, 0:3] = imputer.transform(x[:, 0:3])


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting Decision Tree Classification to the Training set

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
    

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)

## Stochastic Gradient Descent(SGD):

def SGD(x_train,y_train):
    sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
    sgd.fit(x_train, y_train)
    y_pred = sgd.predict(x_test)
    sgd.score(x_train, y_train)
    acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
    return(acc_sgd)
    
# Random forest
    
def RF(x_train,y_train):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)
    random_forest.score(x_train, y_train)
    acc_RF = round(random_forest.score(x_train, y_train) * 100, 2)
    return(acc_RF)
    
#K Nearest Neighbor

def KNN(x_train,y_train):
    knn = KNeighborsClassifier(n_neighbors = 3) 
    knn.fit(x_train, y_train) 
    y_pred = knn.predict(x_test)  
    acc_knn = round(knn.score(x_train, y_train) * 100, 2)
    return(acc_knn)
    
#Gaussian
    
def Gaussian(x_train,y_train):
    gaussian = GaussianNB() 
    gaussian.fit(x_train, y_train) 
    y_pred = gaussian.predict(x_test)  
    acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
    return(acc_gaussian)
    
#Linear Support vector machine:
    
def SVM(x_train,y_train):
    linear_svc = LinearSVC()
    linear_svc.fit(x_train, y_train)
    y_pred = linear_svc.predict(x_test)
    acc_linear_svc = round(linear_svc.score(x_train,y_train) * 100, 2)
    return(acc_linear_svc)
    
    
def DST(x_train,y_train):
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(x_train, y_train)  
    y_pred = decision_tree.predict(x_test)  
    acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
    return(acc_decision_tree)
    
# Selecting the best model:
    

results = pd.DataFrame({
    'Model': ['Stochastic Gradient Decent', 'Random_Forest','KNN', 'Gaussian','Support Vector Machines', 'Decision Tree'],
    'Score': [SGD(x_train,y_train),RF(x_train,y_train),KNN(x_train,y_train),Gaussian(x_train,y_train),SVM(x_train,y_train),DST(x_train,y_train)]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
    
    
    

    











