#!/usr/bin/env python
# coding: utf-8

# # Assignment 3
# All questions are weighted the same in this assignment. This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. All questions are worth the same number of points except question 1 which is worth 20% of the assignment grade.
# 
# **Note**: Questions 2-13 rely on your question 1 answer.

# In[1]:


import pandas as pd
import numpy as np

# Filter all warnings. If you would like to see the warnings, please comment the two lines below.
import warnings
warnings.filterwarnings('ignore')


# ### Question 1
# Load the energy data from the file `assets/Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](assets/Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **Energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable]`
# 
# Convert `Energy Supply` to gigajoules (**Note: there are 1,000,000 gigajoules in a petajoule**). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with parenthesis in their name. Be sure to remove these, e.g. `'Bolivia (Plurinational State of)'` should be `'Bolivia'`.
# 
# Next, load the GDP data from the file `assets/world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `assets/scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries, and the rows of the DataFrame should be sorted by "Rank".*

# In[195]:


import pandas as pd
import numpy as np
# Filter all warnings. If you would like to see the warnings, please comment the two lines below.
import warnings
warnings.filterwarnings('ignore')

def answer_one():

   
    
    ### PREPARSING THE ENERGY USAGE EXCEL FILE
    
    
    Energy = pd.read_excel('assets/Energy Indicators.xls')
    
    #Just for carrying out trials
    new_Energy = Energy.copy()
    
    # Remove header and footer. Including titles and units.
    new_Energy = new_Energy.iloc[17:244]
    # Drops the two first columns
    new_Energy.drop(new_Energy.columns[[0,2]], axis=1,inplace=True)
    # Changing column labels
    new_Energy.rename(columns = {'Unnamed: 1' : 'Country',
                                'Unnamed: 3' : 'Energy Supply',
                                'Unnamed: 4' : 'Energy Supply per Capita',
                                'Unnamed: 5' : '% Renewable'}, inplace = True)

    
    # Replaces values of no information i.e "..." with np.NaN 
    new_Energy['Energy Supply'].replace('...',np.NaN,inplace=True)
    
    # Changes petajoule to gigajoule        
    new_Energy['Energy Supply'] = np.multiply(new_Energy['Energy Supply'],1000000)
    new_Energy.set_index('Country', inplace=True)
    new_Energy.rename(index = {"Republic of Korea": "South Korea",
                                "United States of America": "United States",
                                "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                                "China, Hong Kong Special Administrative Region": "Hong Kong"}, inplace=True)
    
    # Changes all countries having parenthesis. 
    for item in new_Energy.index:
        if item[-1] == ")":
            item_list = item.split('(')
            new_Energy.rename(index = {item : item_list[0].strip()},inplace=True)
    
    
    # Just changing back to normal indexing for now and changing name.
    new_Energy.reset_index(inplace = True)
    new_Energy.index = new_Energy.index+1
    Energy = new_Energy
    
    ### PREPARSING GDP DATA FROM CSV FILE
    
    GDP = pd.read_csv('assets/world_bank.csv')
    
    #Remove header
    GDP = GDP[4:]
    
    #Change name of title
        
    

    GDP.rename(columns = {'Data Source' : 'Country'}, inplace=True)
        
    #Change names
    GDP['Country'].replace({"Korea, Rep." : "South Korea", 
                            "Iran, Islamic Rep.": "Iran",
                            "Hong Kong SAR, China": "Hong Kong"}, inplace=True)
    

    
    
    ################# TRYING TO CHANGE NAME OF A LIST OF COLUMNS #############################
#     GDP.rename(columns = {[GDP.columns[4:]] : [np.arange[1960:(1960+len(GDP.columns[4:]))]]})
    
    
    
#     df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['1', '2', '3'])
    
#     print(df2)
#    df2.rename(columns = {df2.columns[0:]: [4,5,6]}) 
#     df2.columns[0:2] = [4,5,6]
#     print(df2)
    
    ##################################################################################################
    
    
    
    
    ### PREPARSING JOURNAL CONTRIBUTION EXCEL FILE
    
    ScimEn = pd.read_excel('assets/scimagojr-3.xlsx')
    
    
    
    ### JOINING THE TABLES
    
    #Picks out the last 10 years of GDP data. 
    GDP = pd.concat([GDP[GDP.columns[0]],GDP[GDP.columns[-10:]]], axis=1)
    
    merge1 = pd.merge(Energy, GDP, how='inner', on='Country')
    
    # Only takes the top 15 countries from ScimEn according to task. 
    df = pd.merge(merge1, ScimEn[:15], how='inner', on='Country')
    
        
    df.set_index('Country', inplace=True)
    
    
    # Making sure the names and the order of the columns fits the description
    df.rename(columns = {'Unnamed: 50':'2006', 'Unnamed: 51':'2007', 'Unnamed: 52':'2008',
               'Unnamed: 53':'2009', 'Unnamed: 54':'2010', 'Unnamed: 55':'2011',
               'Unnamed: 56':'2012', 'Unnamed: 57':'2013','Unnamed: 58': '2014', 
               'Unnamed: 59':'2015'}, inplace = True)
    
    df = df.reindex(columns=['Rank', 'Documents', 'Citable documents', 'Citations', 
                               'Self-citations','Citations per document', 'H index', 
                               'Energy Supply', 'Energy Supply per Capita','% Renewable',
                               '2006', '2007', '2008', '2009', '2010', '2011', '2012', 
                               '2013', '2014', '2015'])


    
    return df.astype('float')
    

answer_one()
    #raise NotImplementedError()


# In[2]:


assert type(answer_one()) == pd.DataFrame, "Q1: You should return a DataFrame!"

assert answer_one().shape == (15,20), "Q1: Your DataFrame should have 20 columns and 15 entries!"


# In[3]:


# Cell for autograder.


# ### Question 2
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[16]:


get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>\n</svg>')


# In[196]:


def answer_two():
       
    ### PREPARSING THE ENERGY USAGE EXCEL FILE
    
    
    Energy = pd.read_excel('assets/Energy Indicators.xls')
    
    #Just for carrying out trials
    new_Energy = Energy.copy()
    
    # Remove header and footer. Including titles and units.
    new_Energy = new_Energy.iloc[17:244]
    # Drops the two first columns
    new_Energy.drop(new_Energy.columns[[0,2]], axis=1,inplace=True)
    # Changing column labels
    new_Energy.rename(columns = {'Unnamed: 1' : 'Country',
                                'Unnamed: 3' : 'Energy Supply',
                                'Unnamed: 4' : 'Energy Supply per Capita',
                                'Unnamed: 5' : '% Renewable'}, inplace = True)

    
    # Replaces values of no information i.e "..." with np.NaN 
    new_Energy['Energy Supply'].replace('...',np.NaN,inplace=True)
    
    # Changes petajoule to gigajoule        
    new_Energy['Energy Supply'] = np.multiply(new_Energy['Energy Supply'],1000000)
    new_Energy.set_index('Country', inplace=True)
    new_Energy.rename(index = {"Republic of Korea": "South Korea",
                                "United States of America": "United States",
                                "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                                "China, Hong Kong Special Administrative Region": "Hong Kong"}, inplace=True)
    
    # Changes all countries having parenthesis. 
    for item in new_Energy.index:
        if item[-1] == ")":
            item_list = item.split('(')
            new_Energy.rename(index = {item : item_list[0].strip()},inplace=True)
    
    
    # Just changing back to normal indexing for now and changing name.
    new_Energy.reset_index(inplace = True)
    new_Energy.index = new_Energy.index+1
    Energy = new_Energy
    
    ### PREPARSING GDP DATA FROM CSV FILE
    
    GDP = pd.read_csv('assets/world_bank.csv')
    
    #Remove header
    GDP = GDP[4:]
    
    #Change name of title
        
    

    GDP.rename(columns = {'Data Source' : 'Country'}, inplace=True)
        
    #Change names
    GDP['Country'].replace({"Korea, Rep." : "South Korea", 
                            "Iran, Islamic Rep.": "Iran",
                            "Hong Kong SAR, China": "Hong Kong"}, inplace=True)
    

    
    
    ### PREPARSING JOURNAL CONTRIBUTION EXCEL FILE
    
    ScimEn = pd.read_excel('assets/scimagojr-3.xlsx')
    
    
    
    ### JOINING THE TABLES
    
    #Picks out the last 10 years of GDP data. 
    GDP = pd.concat([GDP[GDP.columns[0]],GDP[GDP.columns[-10:]]], axis=1)
    
    merge1 = pd.merge(Energy, GDP, how='inner', on='Country')
    df = pd.merge(merge1, ScimEn, how='inner', on='Country')
    
  
    #The total of entires lost must be the total of entires from all the different regions,
    #minus three times the inner join, i.e the size of the joint set. This answer is:
    
    return GDP.shape[0] + Energy.shape[0]+ ScimEn.shape[0] - 3*df.shape[0]
    
   # raise NotImplementedError()


# In[197]:


assert type(answer_two()) == int, "Q2: You should return an int number!"


# ### Question 3
# What are the top 15 countries for average GDP over the last 10 years?
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[198]:


def answer_three():
    df = answer_one()
    
    
    # Takes out only the GDP values and computes the mean and sorts in descending order.
    return df.iloc[:,10:].mean(axis=1).sort_values(ascending=False)

#     raise NotImplementedError()


# In[199]:


assert type(answer_three()) == pd.Series, "Q3: You should return a Series!"


# ### Question 4
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[200]:


def answer_four():
    df = answer_one()
    avg_GDP_order = answer_three()
    
    #Takes the order from avg_GDP_order to locate the values from the whole dataframe given from answer_one()
    return df.loc[avg_GDP_order.index[5]]['2015'] - df.loc[avg_GDP_order.index[5]]['2006']
#    raise NotImplementedError()


# In[201]:


# Cell for autograder.


# ### Question 5
# What is the mean energy supply per capita?
# 
# *This function should return a single number.*

# In[202]:


def answer_five():
    df = answer_one()
    return df['Energy Supply per Capita'].mean()


    #raise NotImplementedError()


# In[203]:


# Cell for autograder.


# ### Question 6
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[204]:


def answer_six():
    df = answer_one()
    df = df["% Renewable"].sort_values()
    return (df.index[-1], df[-1]) 
   # raise NotImplementedError()


# In[205]:


assert type(answer_six()) == tuple, "Q6: You should return a tuple!"

assert type(answer_six()[0]) == str, "Q6: The first element in your result should be the name of the country!"


# ### Question 7
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[230]:


def answer_seven():
    df = answer_one()
    
    # Create the sorted ration of self-citations to total citations
    df["Ratio of Self-Citations to Total Citations"] = (df['Self-citations']/df['Citations'])
    # Takes out the specific list in order to do sorting and such. 
    sorted_ratio = df["Ratio of Self-Citations to Total Citations"].sort_values()
    
    #Could also have sorted on the entire DateFrame, using the ratio column as input. 
    
    return (sorted_ratio.index[0],sorted_ratio[0])
    
answer_seven()
#     raise NotImplementedError()


# In[231]:


assert type(answer_seven()) == tuple, "Q7: You should return a tuple!"

assert type(answer_seven()[0]) == str, "Q7: The first element in your result should be the name of the country!"


# ### Question 8
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return the name of the country*

# In[208]:


def answer_eight():
    df = answer_one()
    
    df['Population'] = df['Energy Supply']/df['Energy Supply per Capita']
    return (df['Population'].sort_values(ascending=False)).index[2]
    
    #raise NotImplementedError()


# In[209]:


assert type(answer_eight()) == str, "Q8: You should return the name of the country!"


# ### Question 9
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[211]:


def answer_nine():
    import matplotlib as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    df = answer_one()
    df = df.astype('float')
    
    df['Citable docs per Capita'] = (df['Citable documents'])/(df['Energy Supply']/df['Energy Supply per Capita'])
    
    s1 = df['Citable docs per Capita']
    s2 = df['Energy Supply per Capita']
    
    df.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])    
    return s1.corr(s2,method='pearson')

answer_nine()
#     raise NotImplementedError()


# In[187]:


def plot9():
    import matplotlib as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# In[176]:


assert answer_nine() >= -1. and answer_nine() <= 1., "Q9: A valid correlation should between -1 to 1!"


# ### Question 10
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[243]:


def answer_ten():
    df = answer_one()    
    df['HighRenew'] = (df['% Renewable'] > df['% Renewable'].median()).astype('int64')
    return df['HighRenew'].sort_values()

answer_ten()
#     raise NotImplementedError()


# In[244]:


assert type(answer_ten()) == pd.Series, "Q10: You should return a Series!"


# ### Question 11
# Use the following dictionary to group the Countries by Continent, then create a DataFrame that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[350]:


def answer_eleven():
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    df = answer_one()
    df['Population'] = df['Energy Supply']/df['Energy Supply per Capita']

    
    
    # Group and pick out the column population
    group_object = df.groupby(ContinentDict)['Population']
    
#     df2 = pd.DataFrame({'size': group_object['Population'].size()})
    
    pop_df = pd.DataFrame({'size' : group_object.size(),
                        'sum'  : group_object.sum(), 
                        'mean' : group_object.mean(),
                        'std'  : group_object.std()})
    
    
    return pop_df
    
    

answer_eleven()
    
    
#     raise NotImplementedError()


# In[314]:


assert type(answer_eleven()) == pd.DataFrame, "Q11: You should return a DataFrame!"

assert answer_eleven().shape[0] == 5, "Q11: Wrong row numbers!"

assert answer_eleven().shape[1] == 4, "Q11: Wrong column numbers!"


# ### Question 12
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a Series with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[401]:


def answer_twelve():
    ContinentDict  = {'China':'Asia', 
              'United States':'North America', 
              'Japan':'Asia', 
              'United Kingdom':'Europe', 
              'Russian Federation':'Europe', 
              'Canada':'North America', 
              'Germany':'Europe', 
              'India':'Asia',
              'France':'Europe', 
              'South Korea':'Asia', 
              'Italy':'Europe', 
              'Spain':'Europe', 
              'Iran':'Asia',
              'Australia':'Australia', 
              'Brazil':'South America'}
    df = answer_one()
    
    # Creates the two new columns, cutting into 5 groups (% of renewable + continents)
    df['bins'] = pd.cut(df['% Renewable'], 5)
          
    for key in ContinentDict.keys():
        df.at[key,'Continent'] = ContinentDict[key]

    
    # Groups a multiindex thingy
    df = df.set_index(['Continent', 'bins'])    
    return df.groupby(level=(0,1)).size()

answer_twelve()
    
    
    
# raise NotImplementedError()


# In[402]:


assert type(answer_twelve()) == pd.Series, "Q12: You should return a Series!"

assert len(answer_twelve()) == 9, "Q12: Wrong result numbers!"


# ### Question 13
# Convert the Population Estimate series to a string with thousands separator (using commas). Use all significant digits (do not round the results).
# 
# e.g. 12345678.90 -> 12,345,678.90
# 
# *This function should return a series `PopEst` whose index is the country name and whose values are the population estimate string*

# In[423]:


def answer_thirteen():
    df = answer_one()
    df['Population'] = (df['Energy Supply']/df['Energy Supply per Capita'])
    
#     df['Population']=df['Population'].astype(str).replace(r"(\d{3})(\d+)", r"\1,\2", regex=True)
    df['Population'] = df.apply(lambda x: "{:,}".format(x['Population']), axis=1)
        
    return df['Population']

answer_thirteen()
        
        
        
#     raise NotImplementedError()


# In[424]:


assert type(answer_thirteen()) == pd.Series, "Q13: You should return a Series!"

assert len(answer_thirteen()) == 15, "Q13: Wrong result numbers!"


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[ ]:


def plot_optional():
    import matplotlib as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")

