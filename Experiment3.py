#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 19:56:24 2020

@author: shehnazislam
"""

import numpy as np # mathematical operations and algebra
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt # visualization library
import seaborn as sns # Fancier visualizations
#import statistics # fundamental stats package
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import scipy.stats as stats # to calculate chi-square test stat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,confusion_matrix,plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import AdaBoostClassifier


# To view full dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None)

# read data and add the column names
rawData = pd.read_excel('diabetes.xlsx', names=['Age','Gender', 
                    'Polyuria', 'Polydipsia', 'SuddenWeightLoss', 'Weakness', 'Polyphagia', 'GenitalThrush', 
                    'VisualBlurring', 'Itching', 'Irritability', 'DelayedHealing', 'PartialParesis', 'MuscleStiffness', 
                    'Alopecia', 'Obesity', 'Class'])

diabetes = rawData

#Map text into values
diabetes['Class']=diabetes['Class'].map(lambda s:1 if s=='Positive' else 0)
diabetes['Gender']=diabetes['Gender'].map(lambda s:1 if s=='Male' else 0)
diabetes['Polyuria']=diabetes['Polyuria'].map(lambda s:1 if s=='Yes' else 0)
diabetes['Polydipsia']=diabetes['Polydipsia'].map(lambda s:1 if s=='Yes' else 0)
diabetes['SuddenWeightLoss']=diabetes['SuddenWeightLoss'].map(lambda s:1 if s=='Yes' else 0)
diabetes['Weakness']=diabetes['Weakness'].map(lambda s:1 if s=='Yes' else 0)
diabetes['Polyphagia']=diabetes['Polyphagia'].map(lambda s:1 if s=='Yes' else 0)
diabetes['VisualBlurring']=diabetes['VisualBlurring'].map(lambda s:1 if s=='Yes' else 0)
diabetes['Irritability']=diabetes['Irritability'].map(lambda s:1 if s=='Yes' else 0)
diabetes['PartialParesis']=diabetes['PartialParesis'].map(lambda s:1 if s=='Yes' else 0)
diabetes['MuscleStiffness']=diabetes['MuscleStiffness'].map(lambda s:1 if s=='Yes' else 0)
diabetes['Alopecia']=diabetes['Alopecia'].map(lambda s:1 if s=='Yes' else 0)
diabetes['Itching']=diabetes['Itching'].map(lambda s:1 if s=='Yes' else 0)
diabetes['DelayedHealing']=diabetes['DelayedHealing'].map(lambda s:1 if s=='Yes' else 0)
diabetes['Obesity']=diabetes['Obesity'].map(lambda s:1 if s=='Yes' else 0)
diabetes['GenitalThrush']=diabetes['GenitalThrush'].map(lambda s:1 if s=='Yes' else 0)

#At 1% significant level, we are going to drop Itching, DelayedHealing, Obesity, And GenitalThrush since it does not strongly correlated with the Target class as per Chisquare test.

diabetes.drop(['Itching','DelayedHealing','Obesity','GenitalThrush'], axis=1, inplace=True)

#Due to the diffences in symptoms, we are going to divide Age into 3 groups: below 18,18-45, above 45
def age_func(age):
    if age < 18:
        return 'below_18'
    elif age>=18 and age<45:
        return '18_45'
    else:
        return 'above_45';
    
diabetes['Age']=diabetes['Age'].apply(age_func)

df=pd.get_dummies(diabetes)

y=df.pop('Class')
X=df

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)


def result_generator(classifier,name,X_train,X_test,y_train,y_test):
    score=[]
    classifier.fit(X_train, y_train)
    y_pred=classifier.predict(X_test)
    plot_confusion_matrix(classifier, X_test, y_test,cmap=plt.cm.Blues)
    plt.show()
    roc_score_result(y_test,y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    score=[name,accuracy_score(y_test,y_pred),metrics.auc(fpr, tpr),1-accuracy_score(y_test,y_pred)]
    return score


 
def roc_score_result(y_test,y_pred):
    fpr,tpr,_=roc_curve(y_test,y_pred)
    plt.plot(fpr,tpr,color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_score(y_test,y_pred))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('Receiver operating characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    


result=[]
#DecisionTree
result.append(result_generator(DecisionTreeClassifier(random_state=0),'Decision Tree',X_train,X_test,y_train,y_test))

#KNN
result.append(result_generator(KNeighborsClassifier(n_neighbors=3),'KNN',X_train,X_test,y_train,y_test))

#AdaBoost

result.append(result_generator(AdaBoostClassifier(random_state=0),'AdaBoost',X_train,X_test,y_train,y_test))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
result.append(result_generator(clf,'RF',X_train,X_test,y_train,y_test))

result_df=pd.DataFrame(result,columns=['Algorithm','Accuracy','AUC','Error_rate'])










