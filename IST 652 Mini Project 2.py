#!/usr/bin/env python
# coding: utf-8

# In[80]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px


# In[81]:


#reading in our unstructured json file
my_df= pd.read_json("C:/Users/harsh/Downloads/IST 659 Mini Project 2 Final/My_dataset_netflix_json.json")


# In[82]:


#checking
my_df.head()


# In[83]:


#checking number rows and columns
my_df.shape


# In[84]:


#trying to get name of every column
my_df.columns
#df.info


# In[85]:


#my_df.date_added = pd.to_datetime(my_df.date_added)
#my_df["date_added"].fillna("NA", inplace = True)


# In[86]:


df=df.fillna('Not specified')
my_df.isnull().sum()
#no null values


# In[87]:


#checking number of unique values in every column
my_df.nunique(axis=0)


# In[88]:


#for finding top 5 directors
director_name=pd.DataFrame()
director_name=my_df['director'].str.split(',',expand=True).stack()
director_name=director_name.to_frame()
director_name.columns=['Director']
directors=director_name.groupby(['Director']).size().reset_index(name='Total Content')

#ignoring null values

directors=directors[directors.Director !='']

#sorting by number of Total Content by respective director

directors=directors.sort_values(by=['Total Content'],ascending=False)
directorsTop5=directors.head()
directorsTop5=directorsTop5.sort_values(by=['Total Content'])

#saving my file to csv
#directorsTop5.to_csv('directorsTop5.csv')
#directorsTop5.to_csv(r'C:/Users/harsh/Downloads/directorsTop5.csv')

#plotting bar graph showing top 5 directors
fig1=px.bar(directorsTop5,x='Total Content',y='Director',title='Top 5 Directors on Netflix')
fig1.show()


# In[89]:


#for top 5 actors
cast_name=pd.DataFrame()

cast_name=my_df['cast'].str.split(',',expand=True).stack()
cast_name=cast_name.to_frame()
cast_name.columns=['Actor']
actors=cast_name.groupby(['Actor']).size().reset_index(name='Total Content')

actors=actors[actors.Actor !='']
actors=actors.sort_values(by=['Total Content'],ascending=False)
actorsTop5=actors.head()

actorsTop5=actorsTop5.sort_values(by=['Total Content'])

actorsTop5.to_csv('directorsTop5.csv')
actorsTop5.to_csv(r'C:/Users/harsh/Downloads/actorsTop5.csv')

fig2=px.bar(actorsTop5,x='Total Content',y='Actor', title='Top 5 Actors on Netflix')
fig2.show()


# In[90]:


# I will try to see proportion of Ratings accross content in our data set
p=my_df.groupby(['rating']).size().reset_index(name='counts')
piechart=px.pie(p,values='counts',names='rating',title='Ratings of different contents on netflix')
piechart.show()


# In[91]:


#I will try to see which type is more movie or TV Show
df1=my_df[['type','release_year']]
df1=df1.rename(columns={"release_year": "Release Year"})
df2=df1.groupby(['Release Year','type']).size().reset_index(name='Total Content')
df2=df2[df2['Release Year']>=2010]
fig3 = px.line(df2, x="Release Year", y="Total Content", color='type',title='Trend of content produced over the years on Netflix')
fig3.show()


# In[92]:


#importing textBlob for sentiment analysis

from textblob import TextBlob

dfx=my_df[['release_year','description']]
dfx=dfx.rename(columns={'release_year':'Release Year'})

# .iterrows() iterate over DataFrame rows as (index, Series) pairs.

for index,row in dfx.iterrows():
    z=row['description']
    testimonial=TextBlob(z)
    p=testimonial.sentiment.polarity
    if p==0:
        sent='Neutral'
    elif p>0:
        sent='Positive'
    else:
        sent='Negative'
    dfx.loc[[index,2],'Sentiment']=sent


dfx=dfx.groupby(['Release Year','Sentiment']).size().reset_index(name='Total Content')

# getting release year after 2010

dfx=dfx[dfx['Release Year']>=2010]

#dfx.to_csv('sentiment.csv')
#dfx.to_csv(r'C:/Users/harsh/Downloads/sentiment.csv')

fig4 = px.bar(dfx, x="Release Year", y="Total Content", color="Sentiment", title="Sentiment of content on Netflix")
fig4.show()

#a lot of content in 2018, majority positive sentiment in accross all years


# In[ ]:




