#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Import libraries

import pandas as pd
import seaborn as sns
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure


get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) # adjust the configuration of plots we will create

#read in the data
df = pd.read_csv(r'G:\datasets\movies.csv')


# In[25]:


# look at the data
df.head()


# In[26]:


# lets see if there is any missing data
df = pd.read_csv(r'G:\datasets\movies.csv')
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,pct_missing))
  


# In[28]:


# types of data types
df.dtypes


# In[6]:


# create correct year column

import pandas as pd
df = pd.read_csv(r'G:\datasets\movies.csv')

df['released'] = df['released'].astype(str).str[:4]
df['year correct'] = df['year'].astype(str).str[:4]


# In[5]:


df.head()


# In[13]:


df = df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[9]:


pd.set_option('display.max_rows',None)
df


# In[10]:


df.drop_duplicates()


# In[15]:


# scatter plot with budget vs gross

import matplotlib.pyplot as plt
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('budget vs gross earnings')
plt.xlabel('Gross earnings')
plt.ylabel('Budget for film')

plt.show()


# In[14]:


df.head()


# In[19]:


import seaborn as sns
sns.regplot(x='budget',y='gross',data = df, scatter_kws={"color":"red"} , line_kws={"color":"blue"})


# In[20]:


# lets start looking for coorelation


# In[27]:


df.corr(method = 'pearson') # pearson , kendall, spearman


# In[30]:


correaltion_matrix = df.corr(method = 'pearson')

sns.heatmap(correaltion_matrix, annot = True)

plt.title('Correlation matrix for number')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[35]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
               df_numerized[col_name] = df_numerized[col_name].astype('category')
               df_numerized[col_name] = df_numerized[col_name].cat.codes
    
df_numerized


# In[3]:


# Using factorize - this assigns a random numeric value for each unique categorical value

import pandas as pd
df = pd.read_csv(r'G:\datasets\movies.csv')

df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


correlation_mat = df.apply(lambda x: x.factorize()[0]).corr()

corr_pairs = correlation_mat.unstack()

print(corr_pairs)


# In[8]:


sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)


# In[ ]:





# In[ ]:





# In[9]:


# We can now take a look at the ones that have a high correlation (> 0.5)

strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)


# In[10]:


# Looking at the top 15 compaies by gross revenue

CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[ ]:





# In[ ]:





# In[11]:


df['Year'] = df['released'].astype(str).str[:4]
df


# In[ ]:





# In[ ]:





# In[12]:


df.groupby(['company', 'year'])[["gross"]].sum()


# In[ ]:





# In[ ]:





# In[13]:


CompanyGrossSum = df.groupby(['company', 'year'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company','year'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[ ]:





# In[ ]:





# In[14]:


CompanyGrossSum = df.groupby(['company'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[ ]:





# In[ ]:





# In[15]:


plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




