# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd

df1 = pd.read_csv('data/raw/bbc-text.csv')
df1.head()
# -

df2 = pd.read_csv('data/raw/bbc-news-data.csv', sep="\t")
hold2 = df2[['category', 'content']]
# hold2['content'] = df2['title'] + df2['content']
hold2.columns = ['category', 'text']
hold2.head()

df3 = pd.read_csv('data/raw/learn-ai-bbc/BBC News Train.csv')
df3.head()

hold3 = df3[['Category', 'Text']]
hold3.columns = ['category', 'text']
hold3.head()

import chardet    
rawdata = open('data/raw/Data_Train.csv', 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']
print(charenc)

df4 = pd.read_csv('data/raw/Data_Train.csv', encoding="Windows-1252")
df4.head()

# +
for i in range(4):
    hold = df4[df4['SECTION'] == i]
    if i == 0:
        hold['category'] = 'politics'
    elif i == 1:
        hold['category'] = 'tech'
    elif i == 2:
        hold['category'] = 'entertainment'
    elif i == 3:
        hold['category'] = 'business'
        
    if i == 0:
        hold4 = hold
    else:
        frames = [hold4, hold]
        hold4 = pd.concat(frames)
        
hold4 = hold4[['category', 'STORY']]
hold4.columns = ['category', 'text']
hold4.head()
# -

frames = [df1, hold2, hold3, hold4]
final_df = pd.concat(frames).drop_duplicates().reset_index().drop(['index'], axis=1)

final_df

final_df.to_csv('data/all_data.csv', index=False)

final_df['category'].value_counts()


