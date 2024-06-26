#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv("covid19_Confirmed_dataset.csv")
df


# In[3]:


df1= df.drop(["Lat","Long"], axis=1, inplace =True)


# In[4]:


df.head(5)


# In[5]:


combined_corona_case = df.groupby("Country/Region").sum()


# In[6]:


combined_corona_case.head()


# In[7]:


df1= combined_corona_case.drop(["Province/State"], axis=1, inplace =True)


# In[8]:


combined_corona_case.head()


# In[9]:


combined_corona_case.shape


# In[10]:


combined_corona_case.loc["China"].plot()
combined_corona_case.loc["Afghanistan"].plot()
combined_corona_case.loc["India"].plot()
combined_corona_case.loc["Spain"].plot()
plt.legend()


# In[11]:


combined_corona_case.loc["China"][:3].plot()


# In[12]:


combined_corona_case.loc["China"].diff()


# In[13]:


combined_corona_case.loc["China"].diff().plot()


# In[14]:


combined_corona_case.loc["China"].diff().max()


# In[15]:


countries = list(combined_corona_case.index)
max_infection_rate = []

for c in countries:
    max_infection_rate.append(combined_corona_case.loc[c].diff().max())
combined_corona_case["max_infection_rate"] = max_infection_rate


# In[16]:


combined_corona_case


# In[17]:


corona_data = pd.DataFrame(combined_corona_case["max_infection_rate"])


# In[18]:


corona_data


# In[19]:


happiness_report = pd.read_csv("worldwide_happiness_report.csv")


# In[20]:


happiness_report


# In[21]:


useless_cols = ["Overall rank", "Score", "Generosity", "Perceptions of corruption"]


# In[22]:


happiness_report.drop(useless_cols, axis=1, inplace = True)
happiness_report.head()


# In[23]:


happiness_report.set_index("Country or region", inplace = True)
happiness_report.head()


# In[24]:


corona_data.shape


# In[25]:


happiness_report.shape


# In[26]:


data = corona_data.join(happiness_report, how = "inner")


# In[27]:


data


# In[28]:


data.corr()


# In[29]:


import seaborn as sns



# In[32]:


import numpy as np

# Assuming your data is in a pandas dataframe named 'data'
x = data["GDP per capita"]
y = data["max_infection_rate"]

# Apply log transformation to y-axis data (avoid in-place modification)
log_y = np.log(y.copy())  

# Create scatterplot with keyword argument for y
sns.scatterplot(x=x, y=log_y)


# In[37]:


import seaborn as sns
import numpy as np

# Assuming your data is in a pandas dataframe named 'data'
x = data["GDP per capita"]
y = data["max_infection_rate"]

# Apply log transformation (avoid in-place modification)
log_y = np.log(y.copy())  

# Create regression plot with data, log scale on y-axis, and optional arguments
sns.regplot(data=data, x=x, y=log_y, ci=None, line_kws={"color": "red"})


# In[42]:


# Assuming your data is in a pandas dataframe named 'data'
x = data["Freedom to make life choices"]
y = data["max_infection_rate"]

# Apply log transformation (avoid in-place modification)
log_y = np.log(y.copy())  

# Create regression plot with data, log scale on y-axis, and optional arguments
sns.regplot(data=data, x=x, y=log_y, ci=None, line_kws={"color": "green"})


# In[46]:


# Assuming your data is in a pandas dataframe named 'data'
x = data["Healthy life expectancy"] 
y = data["max_infection_rate"]

# Apply log transformation (avoid in-place modification)
log_y = np.log(y.copy())  

# Create regression plot with data, log scale on y-axis, and optional arguments
sns.regplot(data=data, x=x, y=log_y, ci=None, line_kws={"color": "green"})


# In[ ]:




