# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:22:09 2024

@author: DELL
"""

import pandas as pd
import numpy as np

dataset= pd.read_csv("/Users/DELL/Desktop/AI project/Disease_symptom.csv")

dataset=dataset.drop_duplicates()
x=dataset.iloc[:,dataset.columns!='Outcome Variable'].values
y=dataset.iloc[:,9].values


# To transfer the strings to numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
x[:,1]=labelencoder_x.fit_transform(x[:,1])
x[:,2]=labelencoder_x.fit_transform(x[:,2])
x[:,3]=labelencoder_x.fit_transform(x[:,3])
x[:,4]=labelencoder_x.fit_transform(x[:,4])
x[:,5]=labelencoder_x.fit_transform(x[:,5])
x[:,6]=labelencoder_x.fit_transform(x[:,6])
x[:,7]=labelencoder_x.fit_transform(x[:,7])
x[:,8]=labelencoder_x.fit_transform(x[:,8])

LabelEncoder_Y=LabelEncoder()
y=LabelEncoder_Y.fit_transform(y)

# spliting to training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.4,random_state=0)

# training 
from sklearn.preprocessing import StandardScaler
Sc_x =StandardScaler()
x_train=Sc_x.fit_transform(x_train)
x_test=Sc_x.transform(x_test)

#importing the naive bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
#importing the decison tree model
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
y_predtree=dtc.fit(x_train, y_train).predict(x_test)

#importing the neural network model
from sklearn.neural_network import MLPClassifier
nlp=MLPClassifier()
ypred= nlp.fit(x_train, y_train).predict(x_test)

from sklearn.metrics import accuracy_score
#To find the accuracy with the neural network model
acc_nn=accuracy_score(y_test, ypred)
#To find the accuracy with the naive bayes model
acc_nb=accuracy_score(y_test, y_pred)
# To find the accuracy with the decision tree model
acctree=accuracy_score(y_test, y_predtree)

from sklearn.metrics import confusion_matrix
cm_nn =confusion_matrix(y_test,ypred)
cm_nb =confusion_matrix(y_test,y_pred)






