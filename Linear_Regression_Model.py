#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pandas import Series 


# In[22]:


hp = pd.read_csv(r'C:\Users\kkavy\Desktop\Python\Machine_Learning\Linear_Regression_Home_Prices_Data.csv')


# In[23]:


hp


# In[30]:


plt.scatter(hp.area,hp.price, color = 'black', marker ='+')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')


# In[51]:


reg = linear_model.LinearRegression()


# In[52]:


reg.fit(hp[['area']],hp.price)


# In[66]:


reg.predict([[3300]])


# In[55]:


reg.coef_


# In[56]:


reg.intercept_


# In[58]:


d = pd.read_csv(r'C:\Users\kkavy\Desktop\Python\Machine_Learning\areas.csv')
d


# In[60]:


p = reg.predict(d)


# In[61]:


d['prices'] = p


# In[62]:


d


# In[65]:


d.to_csv(r'C:\Users\kkavy\Desktop\Python\Machine_Learning\Prediction.csv', index= False)


# In[72]:


df= pd.read_csv(r'C:\Users\kkavy\Desktop\Python\Machine_Learning\homeprices.csv')


# In[73]:


df


# In[79]:


import math
Med = math.floor(df.bedrooms.median())
Med


# In[81]:


df.bedrooms = df.bedrooms.fillna(Med)
df

reg = linear_model.LinearRegression() 
# In[82]:


reg.fit(df[['area','bedrooms','age']],df.price)


# In[83]:


reg.coef_


# In[84]:


reg.intercept_


# In[85]:


reg.predict([[3000,3,40]])

