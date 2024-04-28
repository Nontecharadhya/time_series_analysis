#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\Abhi\\downloads')


# In[4]:


df =pd.read_csv('temperature.csv')


# In[5]:


df.head()


# In[6]:


df.rename(columns={'Daily minimum temperatures':'min_temp'},inplace =True)


# In[7]:


df.head()


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[13]:


df.describe()


# In[11]:


df['Date'] =pd.to_datetime(df['Date'])


# In[12]:


df['min_temp'] =pd.to_numeric(df['min_temp'],errors='coerce')


# In[14]:


df.head()


# In[14]:


df_81 =df.loc[:'1981-12-31']
df_81


# In[15]:


sns.lineplot(data=df,x=df.min_temp,y=df.Date.dt.year)


# In[17]:


# filter data by yearly
yearly_temp=df.groupby(df['Date'].dt.year)['min_temp'].mean()
yearly_temp

# reset the index number
yearly_temp =yearly_temp.reset_index()
yearly_temp =yearly_temp.rename(columns={'Date':'Year','min_temp':'Temperature'})
yearly_temp


# In[18]:


yearly_temp['Temperature'].plot(figsize=(20, 5))  # Set the figure size
plt.title('Temperature')  # Set the title of the plot
plt.show()


# # ADF -Augmented Dickey -Fuller Test

# In[19]:


df['rollMean'] =df.min_temp.rolling(window=12).mean()
df['rollStd'] =df.min_temp.rolling(window=12).std()


# In[23]:


plt.figure(figsize=(10,6))
sns.lineplot(data =df,x=df.Date,y=df.min_temp)


# In[20]:


plt.figure(figsize=(10,6))
sns.lineplot(data =df,x=df.Date,y=df.min_temp)
sns.lineplot(data=df,x=df.Date,y=df.rollMean)
sns.lineplot(data=df,x=df.Date,y=df.rollStd)


# In[24]:


from statsmodels.tsa.stattools import adfuller
adfTest =adfuller(df['min_temp'],autolag='AIC')
adfTest[0:4]
stats=pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used','no of observation used'])
stats


# In[22]:


df.isnull().sum()


# In[23]:


df.dropna(inplace=True)


# In[25]:


for key,values in adfTest[4].items():
    print('criticality',key,':',values)


# ### Clearly we can see that the T-statistic value is lower than the criticality value so it jusitfy the condition,
# ### So the data is stationary !

# In[26]:


df.head()


# In[26]:


first_t =df[['min_temp']].copy(deep=True)
first_t['firstdiff'] =first_t['min_temp'].diff()
first_t['12diff'] =first_t['min_temp'].diff(12)
first_t.head()


# In[ ]:





# In[27]:


from statsmodels.tsa.SARIMAX import SARIMAX


# In[28]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[29]:


plot_pacf(first_t['firstdiff'].dropna(),lags=20)


# In[ ]:


# p # 1,q = 3, d = 1


# In[30]:


train=first_t[:round(len(first_t)*3/100)]
test=first_t[round(len(first_t)*3/100):]
train.tail()


# In[31]:


from statsmodels.tsa.arima.model import ARIMA

# Assuming train is your training DataFrame and test is your testing DataFrame
# Assuming 'min_temp' is the column containing your time series data

# Create and fit the ARIMA model
model = ARIMA(train['min_temp'], order=(1, 1, 3))
model_fit = model.fit()

# Make predictions
start_date = test.index[0]
end_date = test.index[-1]
predictions = model_fit.predict(start=start_date, end=end_date)

# Display the predictions
print(predictions)


# In[ ]:




