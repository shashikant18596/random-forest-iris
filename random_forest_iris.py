#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[4]:


iris = load_iris()


# In[5]:


dir(iris)


# In[7]:


iris.feature_names


# In[9]:


iris.data[:2]


# In[10]:


iris.filename


# In[11]:


iris.target


# In[12]:


iris.target_names


# In[23]:


df = pd.DataFrame(iris.data,columns = iris.feature_names)
df.head(2)


# In[24]:


df['target'] = iris.target
df[:2]


# In[27]:


x = df.drop('target',axis = "columns")
x[:2]


# In[30]:


y = df['target']
y[:2]


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
model = RandomForestClassifier(n_estimators=800)
model.fit(x_train,y_train)


# In[50]:


model.predict(x_test)


# In[51]:


model.score(x_test,y_test)


# In[53]:


cm = confusion_matrix(y_test,model.predict(x_test))
cm


# In[57]:


sn.heatmap(cm,annot=True)
plt.xlabel('Prediction')
plt.ylabel('Truth')


# In[ ]:




