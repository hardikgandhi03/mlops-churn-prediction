#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('../raw_data/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[4]:


# Quick view
print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df['Churn'].value_counts())


# In[5]:


# Class balance
sns.countplot(data=df, x='Churn')
plt.title("Class Balance - Churn")
plt.show()

