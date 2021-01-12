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


# To view full dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None)

# read data and add the column names
rawData = pd.read_excel('diabetes.xlsx', names=['Age','Gender', 
                    'Polyuria', 'Polydipsia', 'SuddenWeightLoss', 'Weakness', 'Polyphagia', 'GenitalThrush', 
                    'VisualBlurring', 'Itching', 'Irritability', 'DelayedHealing', 'PartialParesis', 'MuscleStiffness', 
                    'Alopecia', 'Obesity', 'Class'])

rawData.info()

# view the first 5 rows, "head" is the function that return the first "n" rows for the object based on position, by default, "5"
# Notes: index start from "0"
rawData.head()

# last 5 rows
rawData.tail()

# gerneral info
diabetes = rawData
diabetes.info()
diabetes.dtypes

#Missing values
print('---------------- Count ----------------------\n')
print(diabetes.count()) # no missing values(blank/NA)
null = sns.heatmap(diabetes.isnull())
null.set_title("Missing Values Count")
plt.show()

#Outlier for numeric value
diabetes.plot(kind='box', subplots=True, layout=(1,3), sharex=False, sharey=False, figsize=(20,6))
plt.show()

#Unique values for non-numeric values
cols=diabetes.columns
for x in cols:
    print(diabetes[x].value_counts())

#Data Reduction
list=['Gender','Polyuria', 'Polydipsia', 'SuddenWeightLoss', 'Weakness', 'Polyphagia', 'GenitalThrush', 
                    'VisualBlurring', 'Itching', 'Irritability', 'DelayedHealing', 'PartialParesis', 'MuscleStiffness', 
                    'Alopecia','Obesity', 'Class']

p_value_list= []
# Crosstab or contingency or frquency table between class atribute 'class' and all 16 non-numeric attribtess
for x in list:
    print('--------------------------------------'+ x + '---------------------------------\n')
    #stacked bar plot
    ct=pd.crosstab(diabetes['Class'], diabetes[x])
    ct.plot.bar(stacked=True)
    plt.show()
    print(ct)
    print('\n')
    print(stats.chi2_contingency(pd.crosstab(diabetes['Class'], diabetes[x])))
    t_test_results= stats.chi2_contingency(pd.crosstab(diabetes['Class'], diabetes[x]))
    print(t_test_results[1])
    
    p_value_list.append(t_test_results[1])  

for x, y in zip(list, p_value_list):
    print(str(x)+'----------'+str(y))

# At 5% significant level, ChiSquare statistics
# Gender			3.289703730553317e-24
# Polyuria			1.7409117803442155e-51
# Polydipsia			6.1870096408863144e-49
# SuddenWeightLoss			5.969166262549937e-23
# Weakness			4.869843446585542e-08
# Polyphagia			1.1651584346409174e-14
# GenitalThrush			0.016097902991938178
# VisualBlurring			1.7015036753241226e-08
# Irritability			1.7714831493959365e-11
# PartialParesis			1.565289071056334e-22
# MuscleStiffness			0.006939095697923978
# Alopecia			1.9092794963634e-09


# Attributes not correlated to class

# Itching			0.8297483959485009
# DelayedHealing			0.32665993771439944
# Obesity			0.12710799319896815
    
# At 1% significant level ChiSquare statistics
# Gender			3.289703730553317e-24
# Polyuria			1.7409117803442155e-51
# Polydipsia			6.1870096408863144e-49
# SuddenWeightLoss			5.969166262549937e-23
# Weakness			4.869843446585542e-08
# Polyphagia			1.1651584346409174e-14
# VisualBlurring			1.7015036753241226e-08
# Irritability			1.7714831493959365e-11
# PartialParesis			1.565289071056334e-22
# MuscleStiffness			0.006939095697923978
# Alopecia			1.9092794963634e-09


# Attributes not correlated to class

# Itching			0.8297483959485009
# DelayedHealing			0.32665993771439944
# Obesity			0.12710799319896815
# GenitalThrush			0.016097902991938178
    
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

diabetes.head()

#Correlation Matrix
cor_mat=diabetes.corr()
sns.heatmap(cor_mat)
plt.show()

print(cor_mat['Class'].sort_values(ascending=False))

#Gender,Alopecia have negative correlation with diabetes meaning being a female or having no Alopecia is moderately correlated with Diabetes.


#At 1% significant level, we are going to drop Itching, DelayedHealing, Obesity, And GenitalThrush since it does not strongly correlated with the Target class as per Chisquare test.

diabetes.drop(['Itching','DelayedHealing','Obesity','GenitalThrush'], axis=1, inplace=True)


#EDA
diabetes.describe().unstack()

#Numeric Variable :Age
print('\n--------Age------------')
print(diabetes['Age'].unique())
print('\n')
print(diabetes['Age'].value_counts())
diabetes['Age'].value_counts().plot(kind='bar')
plt.show()
#Histogram
sns.distplot(diabetes['Age'], bins=30)
plt.show()
# histgram indicate that there may be a "problem" since there is a lump at the right tail of the bell curve
#let's further explore the Age variable 
# centre tendency, Median = 47.5
print(np.quantile(diabetes["Age"], 0.5))

# spread,  Q1 = 39 , Q3= 57, IQR= 18
age_Q1=np.quantile(diabetes["Age"], 0.25)
print('Q1:',age_Q1)

age_Q3=np.quantile(diabetes["Age"], 0.75)
print('Q3:', age_Q3)

age_IQR= age_Q3-age_Q1
print('IRQ:',age_IQR)

#IRQ tells us that the middle 50% of the age values are in the range of 39 to 57.
diabetes['Age'].plot(kind='box', subplots=True, figsize=(20,6))
plt.show()
# what do you observe? 2 Outliers, slighly dense towards lower values so positively skewed

# note: A symmetrical distribution will have a skewness of 0.
# Skewness is a measure of symmetry, or more precisely, the lack of symmetry
print('-----------Skewness-------------')
print(diabetes.Age.skew(axis = 0, skipna = True))  # Value of skewness only 0.33 which is low and so we can say the distribution of age is approximately symettric.

# Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. 
# That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. 
print('\n-----------Kurtosis-------------')
print(diabetes.kurtosis(skipna = True)) # Value of skewness -0.19 which indicate that the distribution of age is approximately normally distributed.

#qplot for normality
stats.probplot(diabetes.Age, plot=plt)

sns.catplot(x='Class',y='Age', data=diabetes)
#The plot shows that being above 80 is more likely to get diagnosed with diabetes.
# There is a case of 16 year old also got diagnosed with diabetes. 

#According to healthline website,in 2015, adults aged 45 to 64 were the most diagnosed age group for diabetes.
#Middle-aged and older adults are still at the highest risk for developing type 2 diabetes.
#https://www.medicalnewstoday.com/articles/317375
#Based on this article, we are going to divide the Age into 3 ranges, less than 45, 45-65, above 66.


#Plots of non-numerical variables
cat_list=['Gender','Polyuria', 'Polydipsia', 'Weakness', 'Polyphagia',
                    'VisualBlurring', 'Irritability', 'PartialParesis', 'MuscleStiffness', 
                    'Alopecia', 'Class']

for x in cat_list:
    print('\n---------- ' + x + '------------')
    print(diabetes[x].unique())
    print('\n')
    print(diabetes[x].value_counts())
    counts = diabetes[x].value_counts()
    percentage = diabetes[x].value_counts(normalize='true')
    print('\n')
    print(pd.concat([counts,percentage], axis=1, keys=['counts', '%']))
    
    # plot pie chart
    diabetes[x].value_counts().plot(kind='pie', legend=True)
    plt.show()

#Observation: Numer of Male are as twice as number of female.
#Target class: Note here that the positive cases is 120 more than negative case.

#Line 686 # NoDependents 

#Data Transformation
#Typically ype 1 diabetes usually appears during childhood and adolescent
    
#Type 1 diabetes signs and symptoms can appear relatively suddenly and may include:
#Increased thirst
#Frequent urination
#Bed-wetting in children who previously didn't wet the bed during the night
#Extreme hunger
#Unintended weight loss
#Irritability and other mood changes
#Fatigue and weakness
#Blurred vision
    
    
#Signs and symptoms of type 2 diabetes often develop slowly. In fact, you can have type 2 diabetes for years and not know it. Look for:
#Increased thirst
#Frequent urination
#Increased hunger
#Unintended weight loss
#Fatigue
#Blurred vision
#Slow-healing sores
#Frequent infections
#Areas of darkened skin, usually in the armpits and neck
    
#Type1 and Type2 common symptoms:
#Increased thirst
#Frequent Urination
#Increased hunger
#Unintended weight loss
#Fatigue
#Blurred vision

#However, 
#type1 developed irritation and mood changes and weakness while type 2 show delayed healing

#Due to the diffences in symptoms, we are going to divide Age into 3 groups: below 18,18-45, above 45
def age_func(age):
    if age < 18:
        return 'below_18'
    elif age>=18 and age<45:
        return '18_45'
    else:
        return 'above_45';
    
diabetes['Age']=diabetes['Age'].apply(age_func)

diabetes['Age'].value_counts().plot(kind='pie',autopct='%1.1f%%',legend=True)
plt.show()





