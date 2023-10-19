#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
import pandas as pd 
from os import path
from PIL import Image

import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import string
import heapq
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Importing TextBlob
from textblob import TextBlob


# In[8]:


tweets = pd.read_csv("Hurricane_Harvey.csv")

tweets.dropna(inplace=True)

tweets.head()



# In[9]:


#Understanding data types and info
tweets.dtypes


# In[10]:


tweets.info()


# In[11]:


tweets.describe()


# In[12]:


# Let's see the length of the tweets
seq_length = [len(i) for i in ['Tweet']]

pd.Series(seq_length).hist(bins = 25)


# In[13]:


tweets['Tweet']=tweets['Tweet'].str.replace('[#,@,&]', '')
#Remove twitter handlers
tweets['Tweet']=tweets['Tweet'].str.replace('@[^\s]+','')
#Remove digits
tweets['Tweet']=tweets['Tweet'].str.replace(' \d+ ','')
# remove multiple spaces with single space
tweets['Tweet']=tweets['Tweet'].str.replace("http\S+", "")
# remove multiple spaces with single space
tweets['Tweet']=tweets['Tweet'].str.replace('\s+', ' ')
#remove all single characters
tweets['Tweet']=tweets['Tweet'].str.replace(r'\s+[a-zA-Z]\s+', '')


# In[14]:


# Get stopwords
# Define nltk stopwords in english
stop_words = stopwords.words('english')
stop_words.extend(['ha', 'wa', '-'])

# Get a string of tweets 
tweet_text = ",".join(review.lower() for review in tweets.Tweet if 'covid' not in review)

# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=50, 
                      max_words=100, 
                      stopwords=stop_words,
                      scale=5,
                      background_color="white").generate(tweet_text)

# Display the generated image:
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most repeated words in tweets',fontsize=15)
plt.show()


# In[15]:


# lemmatize text column by using a lemmatize function
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text.lower())]


# Initialize the Lemmatizer and Whitespace Tokenizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Lemmatize words
tweets['lemmatized'] = tweets['Tweet'].apply(lemmatize_text)
tweets['lemmatized'] = tweets['lemmatized'].apply(lambda x: [word for word in x if word not in stop_words])

# use explode to expand the lists into separate rows
wf_tweets = tweets.lemmatized.explode().to_frame().reset_index(drop=True)

# plot dfe
sns.countplot(x='lemmatized', data=wf_tweets, order=wf_tweets.lemmatized.value_counts().iloc[:10].index)
plt.xlabel('Most common used words')
plt.ylabel('Frequency [%]')
plt.xticks(rotation=70)


# In[16]:


#Sentiment Analysis with Textblob
tweets['polarity'] = tweets['Tweet'].apply(lambda x: TextBlob(x).polarity)
tweets['subjectivity'] = tweets['Tweet'].apply(lambda x: TextBlob(x).subjectivity)
tweets


# In[17]:


testimonial = TextBlob("hate")
print(testimonial.sentiment)


# In[18]:


# Unit testing
import unittest

class Test_testimonial(unittest.TestCase):
    def test_ti(self):
        self.ti = testimonial.sentiment
        self.assertNotEqual(self.ti, ('polarity==-0.8', 'subjectivity==0.9'))
        
unittest.main(argv=['first-arg-is-ignored'], defaultTest='Test_testimonial', verbosity=2, exit=False)


# In[19]:


# Shows the top 5 tweets with highest polarity and subjectivity scores
tweets.nlargest(5, ['polarity', 'subjectivity'])['Tweet']


# In[20]:


# Shows the top 5 tweets with lowest polarity and subjectivity scores
tweets.nsmallest(5, ['polarity', 'subjectivity'])['Tweet']


# In[21]:


#setting up our sentiment column and counting the posotive negative and neutral 
tweets['sentiment'] = np.where(tweets.polarity > 0, 'positive', 
                                 np.where(tweets.polarity < 0, 'negative', 'neutral'))
tweets.head()
tweets['sentiment'].value_counts()


# In[22]:


plt.figure(figsize=(15,5))
plt.title('Distribution Of Sentiments Across Our Tweets',fontsize=12,fontweight='bold')
sns.kdeplot(tweets['polarity'], label='Polarity', lw=2.5)
sns.kdeplot(tweets['subjectivity'], label='Subjectivity', lw=2.5)
plt.xlabel('Polarity|subjetivity Value', fontsize=10)
plt.ylabel('Density', fontsize=10)
# Display the generated image:

plt.legend()
plt.show()


# In[23]:


plt.figure(figsize=(15,5))
plt.title('Cumulative Distribution Of Sentiments Across Our Tweets',fontsize=12, fontweight='bold')
sns.kdeplot(tweets['polarity'],cumulative=True, label='Polarity',lw=2.5)
sns.kdeplot(tweets['subjectivity'],cumulative=True, label='Subjectivity',lw=2.5)
plt.xlabel('Polarity Value', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.show()


# In[24]:


#VADER
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
print(vader.polarity_scores(tweet_text))


# In[25]:


for i in tweets.index : print (tweets['ID'][i], vader.polarity_scores(tweets['Tweet'][i]))


# In[26]:


#Findings
#Overall suprsingly most sentinments from the tweets were netrual, then positive, the negative. This shows that a great amount of people were able to disply empathy and calmness during the natural diaster. The stakeholder will be pleased to see that the software work properly and completes is tasks.

