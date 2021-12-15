#!/usr/bin/env python
# coding: utf-8

# # Material prediction using ML

# ### Library import

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# ## Data Description
# The material data is taken from different sources such as web pages, question-answer website, news, etc., and processed into structured data with the help of cleansing and deduplication algorithms. Data used in this project consists of 3 material properties i.e density, elasticity and strength. 3 types of materials are considered which are polymers, metals and ceramics. Data collected here are of a total of 950 materials among which 488 are polymers 325 are ceramics and 157 are metals.
# 
# ## Objective
# The objective of this project is to discover patterns in the materials data and then make predictions using machine learning algorithms to answer engineering questions, detect and analyse, predict trends and help solve problems.
# 
# ## Question and answering with data
# The user feed in material properties to the code and the code helps the user to identify to which type of material it belongs. In this project the user feeds in material properties like density, elasticity and strength and finds out whether it is a polymer or ceramic or metal.
# 
# ### Importing material data

# In[2]:


polymer_data=pd.read_excel("D:/ML_project/polymer.xlsx",dtype={'DensityA': np.float64, 'DensityB': np.float64, 'DensityA': np.float64, 'DensityB': np.float64, 'ElasticityA': np.float64, 'ElasticityB': np.float64, 'StrengthA': np.float64, 'StrengthB': np.float64})
polymer_data.head()


# In[3]:


metal_data=pd.read_excel("D:/ML_project/metal.xlsx",dtype={'DensityA': np.float64, 'DensityB': np.float64, 'DensityA': np.float64, 'DensityB': np.float64, 'ElasticityA': np.float64, 'ElasticityB': np.float64, 'StrengthA': np.float64, 'StrengthB': np.float64})
metal_data.head()


# In[4]:


ceramic_data=pd.read_excel("D:/ML_project/ceramic.xlsx",dtype={'DensityA': np.float64, 'DensityB': np.float64, 'DensityA': np.float64, 'DensityB': np.float64, 'ElasticityA': np.float64, 'ElasticityB': np.float64, 'StrengthA': np.float64, 'StrengthB': np.float64})
ceramic_data.head()


# ### Appending all the materials data to one single dataframe

# In[5]:


data=polymer_data.append(metal_data)
data=data.append(ceramic_data)
data=data.reset_index(drop=True)
data


# ### Replacing max and min of material properties with its mean

# In[6]:


data['Density']=data[['DensityA','DensityB']].mean(axis=1)
data['Elasticity']=data[['ElasticityA','ElasticityB']].mean(axis=1)
data['Strength']=data[['StrengthA','StrengthB']].mean(axis=1)
data=data.drop(columns=['DensityA', 'DensityB', 'ElasticityA', 'ElasticityB', 'StrengthA','StrengthB'], axis=1)
data


# ### Normalising the data

# In[7]:


df_min_max_scaled = data.copy()
df_min_max_scaled=df_min_max_scaled.drop(columns=['Material'], axis=1)
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
df_min_max_scaled['Material']=data.drop(columns=['Density', 'Elasticity', 'Strength'], axis=1)
data=df_min_max_scaled
data


# ### Data cleanup

# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


data=data.drop_duplicates(keep="first")
data.duplicated().sum()


# In[11]:


data=data.reset_index(drop=True)
data


# ## Methodology
# In this project we predict material type based on its material properties. From figure below which is known as Ashby's plot we can see that materials when plotted with respect to its density, elasticity and strength form clusters. We can harness this characteristics of materials and use it for classifying materials. 
# ![image.png](attachment:image.png) ![image-2.png](attachment:image-2.png)

# In[12]:


sns.pairplot(data,hue='Material')


# ### Data preparation
# Initially we convert categorical variables into a numeric by using Ordinal encoder. Then data is splitted into training and testing data. 

# In[13]:


from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
data["Material_code"] = ord_enc.fit_transform(data[["Material"]])
data[["Material", "Material_code"]]
data=data.drop(columns=['Material'])
data


# ### Data spliting

# In[14]:


y = data['Material_code']
X = data.drop('Material_code', axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# ### Model fit using logistic regression

# In[15]:


lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr.fit(train_X,train_y)
y_pred = lr.predict(val_X)
precision = metrics.accuracy_score(y_pred,val_y) * 100
print("Accuracy with Logistic Regression: {0:.2f}%".format(precision))


# ### Model fit using Decision tree

# In[16]:


tree = DecisionTreeClassifier()
tree = tree.fit(train_X,train_y)
y_pred = tree.predict(val_X)
precision = metrics.accuracy_score(y_pred,val_y) * 100
print("Accuracy with Decision Tree: {0:.2f}%".format(precision))


# ## Model accuracy enhancement 
# We found the accuracy to be about 94%. Accuracy of a model is condidered good when its more than 95% and this done by using by using random forest algorithm. Random forest adds additional randomness to the model, while growing the trees. Instead of searching for the most important feature while splitting a node, it searches for the best feature among a random subset of features. This results in a wide diversity that generally results in a better model.

# ### Model fit using Random Forest

# In[17]:


model = RandomForestClassifier(random_state=1)
model.fit(train_X, train_y)
pred_y = model.predict(val_X)
accuracy = metrics.accuracy_score(val_y, pred_y) * 100
print("Accuracy with Random Forest: {0:.2f}% ".format(accuracy))


# In[18]:


pickle.dump(model, open('model.pkl','wb'))


# ## Result discussion
# By applying classifier algorithms like logistic regression, decision tree and random forest we were able to train and fit the model. Using logistic regression we were able to achieve 88.26% and 94.84% of accuracy using decision tree which was enhanced to 95.3% by using random forest.

# ## Conclusion
# In this project we built a system can detect patterns and relationships in data, example from web data, where it will predict trends in the material properties. This project can be applied to predict trends, detect anomalies and find the optimum solution to engineering problems.In order to evaluate the performance of the system, the metrics like precision, recall was used. After this, the data is queried for extracting the relevant information. With the help of different machine learning algorithms, the data is processed and patterns are discovered. Prediction is then made to answer the engineering questions.
