#!/usr/bin/env python
# coding: utf-8

# # MILESTONE 4 - Interpretation of Data & Communication of Insights of Data

# ## Topic : Twitter Sentiment Analysis on Covid19 and Depression
# 
# ### Matric No : 17198801/1  (Aimi Nabilah Hassin)
# 
# Link : https://github.com/aimihassin/WQD7005---Data-Mining-Milestones

#     In this milestone, we aim to achieve the below goals:
#         
#         (a) To understand the relationship between Covid19 and depression
#         (b) To identify the sentiment produced during the Covid19 pandemic
#         (c) To identify the most common words tweeted by people during the pandemic crisis
#         (d) To train and test the prediction model in determining the sentiment label 

# ## Introduction
# 
#     The 2019-Corona Virus is a contagious coronavirus that hailed from Wuhan, China. This new strain of the virus has stricken fear in many countries as cities are quarantined and hospitals are overcrowded. The increment of the total cases of this outbreak worldwide results in the declaration of pandemic by WHO in January, 2020. Most countries have enforced lockdown or movement control order as one of the strategies to curb this infectious virus. As people are not allowed to move freely, the mental state and the well-being of human are being at stake. Hence, this project is intended to understand the impact of Covid19 on human's mental health, including depression, anxiety, stress and many more by analysing the sentiments on tweets. 
#   

# ## Materials and Methods
# 
#    ### Data Collection
#        Data from this project was collected through Twitter by using text mining technique namely data scraping. An access via Twitter API has been made to scrape the tweets. Approximately 20k tweets was scraped, starting from 2019-12-31 which is the date where Covid19 was identified in Wuhan.
#    
#    ### Data Preprocessing
#        Tweets can be considered as unstructured data as they are in the form of textual data. To preprocess textual data, many things need to be done. In this project, we select only desired attributes for our analysis i.e. 'location', 'tweetcreatedts', 'text' and 'hashtags'. Note that any duplicate text shall be removed too to avoid redundancy.
#    
#        In the more advanced data preprocessing, we clean the tweets by removing the urls as well as stopwords. Afterwards, we determine the VADER sentiment score on each tweet and label them according to the VADER sentiment label, where 1 indicates positive sentiment and -1 is negative sentiment. Meanwhile, 0 would mean that the tweet has a neutral sentiment. 

#    ##### Basic Data Preprocessing

# In[130]:


#Import all required libraries
import pandas as pd
import numpy as np

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import re
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()


# In[131]:


#Covidepression_tweet dataset
covidep_df = pd.read_csv("covidep_tweetdata.csv")
covidep_df.head()


# In[132]:


covidep_df.drop(['username', 'acctdesc', 'following', 'followers', 'totaltweets', 'usercreatedts', 'retweetcount'], axis = 1, inplace = True)
covidep_df.tail()


# In[133]:


#drop the duplicate (if any)
data = covidep_df.drop_duplicates()
data.tail()


# In[134]:


#export the dataset
export_csv = data.to_csv(r'covidep_tweets.csv')


# ##### More Advanced Data Preprocessing

# In[135]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import re
import networkx as nx
import nltk
nltk.download(['punkt','stopwords'])
from nltk.corpus import stopwords
from nltk import bigrams

from textblob import TextBlob

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
nltk.download('vader_lexicon')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[136]:


df = pd.read_csv("covidep_tweets.csv")


# In[137]:


#Checking for any null values
df.isnull().any().any()


# In[138]:


df.info(null_counts=True)


# In[139]:


new_df = df[df['text'].notnull()]


# In[140]:


new_df.info(null_counts=True)


# In[141]:


new_df.isnull().any().any()


# In[142]:


#Removing stopwords
stop = stopwords.words('english')

new_df['clean_tweet'] = new_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
new_df.head()


# In[143]:


#Analysing tweets for VADER sentiment analysis
new_df['vader_score'] = new_df['clean_tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])


# In[144]:


#Labeling the sentiment analysis
positive_num = len(new_df[new_df['vader_score'] > 0.05]) 
negative_num = len(new_df[new_df['vader_score'] < -0.05])
neutral_num = len(new_df[new_df['vader_score'].between (-0.05, 0.05)])

new_df['vader_sentiment_label'] = 0
new_df.loc[new_df['vader_score'] > 0.05, 'vader_sentiment_label'] = 1
new_df.loc[new_df['vader_score'] < -0.05, 'vader_sentiment_label'] = -1
new_df.loc[new_df['vader_score'].between (-0.05, 0.05), 'vader_sentiment_label'] = 0
new_df.head(10)


# In[145]:


#Drop any unnecessary attributes (if any)
new_df = new_df[['location', 'tweetcreatedts', 'clean_tweet', 'vader_score', 'vader_sentiment_label']]
new_df.head()


# In[146]:


#Adding a column to describe the sentiment
def label_sentiment (row):
   if row['vader_sentiment_label'] == 1 :
      return 'positive'
   if row['vader_sentiment_label'] == 0 :
      return 'neutral'
   if row['vader_sentiment_label'] == -1:
      return 'negative'
   return 'Other'


# In[147]:


new_df['sentiment'] = new_df.apply (lambda row: label_sentiment(row), axis=1)
new_df.head()


# In[148]:


#Export to csv
new_df.to_csv('covidep_processed_final.csv')


# In[149]:


#Positive sentiment tweets
print('5 random tweets with the highest positive sentiment polarity: \n')
cl = new_df.loc[new_df.vader_sentiment_label == 1, ['clean_tweet']].sample(5).values
for c in cl:
    print(c[0])


# In[150]:


#Neutral sentiment tweets
print('5 random tweets with the most neutral sentiment(zero) polarity: \n')
cl = new_df.loc[new_df.vader_sentiment_label == 0, ['clean_tweet']].sample(5).values
for c in cl:
    print(c[0])


# In[151]:


#Negative sentiment tweets
print('5 tweets with the most negative polarity: \n')
cl = new_df.loc[new_df.vader_sentiment_label == -1, ['clean_tweet']].sample(5).values
for c in cl:
    print(c[0])


# In[152]:


#Removing URLs
def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
all_tweets_no_urls = [remove_url(tweet) for tweet in new_df['clean_tweet']]
all_tweets_no_urls[:5]


# In[153]:


#Changing font to lowercase and split the tweets
#lower_case = [word.lower() for word in new_df['clean_tweet']]
sentences = new_df['clean_tweet']
all_tweets_no_urls[0].split()
words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
words_in_tweet[:2]


# In[154]:


#Count the common words
# List of all words
all_words_no_urls = list(itertools.chain(*words_in_tweet))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(20)


# In[155]:


#turn into dataframe
clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(20),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()


# In[156]:


##### if too many stopwords
stop_words = set(stopwords.words('english'))
# Remove stop words from each tweet list of words
tweets = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

tweets[0]


# In[157]:


all_words = list(itertools.chain(*tweets))  
counts_twit = collections.Counter(all_words)  
counts_twit.most_common(50)


# In[158]:


collection_words = ['amp', 'to', 'my', 'how', 'forbes', 'may', 'us', 'also', 'via', 'one', 'due', 'many', 'get']
tweets_nc = [[w for w in word if not w in collection_words]
                 for word in tweets]


# In[159]:


# Flatten list of words in clean tweets
all_words_nc = list(itertools.chain(*tweets_nc))

# Create counter of words in clean tweets
counts_twit_nc = collections.Counter(all_words_nc)

counts_twit_nc.most_common(20)


# In[160]:


clean_tweets = pd.DataFrame(counts_twit_nc.most_common(20),
                             columns=['words', 'count'])
clean_tweets


# In[161]:


clean_tweets.to_csv(r'covidep_clean_tweets.csv', index=False, header=True)


# In[162]:


from nltk import bigrams

# Create list of lists containing bigrams in tweets
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nc]

# View bigrams for the first tweet
terms_bigram[0]
# Flatten list of bigrams in clean tweets
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)

bigram_counts.most_common(20)
bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                         columns=['bigram', 'count'])
bigram_df


#    ## Exploratory Data Analysis and Visualization
#    Based on the VADER sentiment score and label results above, we able to illustrate the data for better visuality.

# In[163]:


#Distribution of sentiment polarity
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px


hist_fig = px.histogram(x=new_df['vader_score'], nbins=50, title='Distribution of Sentiment Polarity',

                   labels={'x':'sentiment polarity'})

hist_fig.show()


#     From the diagram above, we can observe that the sentiment polarity is skewed to the left, which indicates that most tweets have negative sentiment. It is also noted that the highest tweet count is at sentiment polarity of -0.95 to -0.90, which has  approximately 317 tweets. Meanwhile, neutral sentiment contains the lowest distribution, behind the positive sentiment tweets.

# In[164]:


#Percentage of sentiment polarity
fig, ax = plt.subplots(figsize=(8, 8))

counts = new_df.vader_sentiment_label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")
ax.set_title("Percentage of Sentiment Polarity")

plt.show()


#     From the barplot above, we can distinguish clearly the percentage of sentiment polarity whereby 76% of tweets have negative sentiments, followed by 20% of positive sentiments and merely 4% of the tweets are neutral.

# In[165]:


#Count of sentiment polarity
fig, ax = plt.subplots(figsize=(8, 8))

counts = new_df.vader_sentiment_label.value_counts(normalize=None)
sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Count")
ax.set_title("Sentiment Polarity Count")

plt.show()


# In[166]:


#Visualization of the most common word before removing stopwords
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(
    y=clean_tweets_no_urls['words'],
    x=clean_tweets_no_urls['count'],
    orientation='h',
    marker=dict(
        color='orange')
    ))
fig.update_layout(
    title='Top 20 words found in tweets before removing stop words (including all words)',
    ),
fig.show()


#     Before removing the stopwords, 'i', 'to', 'amp' are also included in the Top 20 words found in the tweets. Though these words are considered as top words but it does not give any meaningful idea or end-product to us. Therefore, we have to remove them by categorising them as stopwords. 
#     
#     The after-removing stopwords has yielded to the below horizontal barplot where covid19 (2378) maintains as the highest word count found in the tweet, followed by stress (1575), depression (1490) and many more.

# In[167]:


#Visualization of the most common word after removing stopwords

fig = go.Figure()
fig.add_trace(go.Bar(
    y=clean_tweets['words'],
    x=clean_tweets['count'],
    orientation='h',
    marker=dict(
        color='indianred')
    ))
fig.update_layout(
    title='Top 20 words found in tweets after removing stop words (including all words)',
    ),
fig.show()


# #### Creating WordCloud

# In[168]:


import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from IPython.display import Image as im

mask = np.array(Image.open('C:/Users/ASUS/OneDrive - 365.um.edu.my/MSc Data Science/Sem 3/Data Mining/Assignment/coronavirus.jpg'))

wc = WordCloud(background_color="white", max_words=2000, mask=mask)
clean_string = ','.join(clean_tweets.words)
wc.generate(clean_string)

f = plt.figure(figsize=(80,80))
f.add_subplot(1,2, 1)
plt.imshow(mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.title('Original Stencil', size=40)
plt.axis("off")
f.add_subplot(1,2, 2)
plt.imshow(wc, interpolation='bilinear')
plt.title('Twitter Generated Cloud', size=40)
plt.axis("off")
plt.show()


#     The wordcloud above illustrates the most common word found in the tweets. The biggest font indicates the highest number of the word count in the tweets, while the smaller font is proportionally to the word count in the tweets.

# In[169]:


#Save as image
wc.to_file("wordcloud.png")


# ## Modeling
#      

# In[170]:


import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords


# In[171]:


dff = pd.read_csv('covidep_processed_final.csv')
dff.head()


# In[172]:


# Spliting target variable and independent variables
X = dff.drop(['vader_sentiment_label'], axis = 1)
y = dff['vader_sentiment_label']


# In[173]:


twit = dff.clean_tweet
twit.head()


# ### TF-IDF
# 
#        Term frequency = (Number of Occurrences of a word)/(Total words in the document)
# 
#        IDF(word) = Log((Total number of documents)/(Number of documents containing the word))

# In[174]:


#Finding TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(twit).toarray()


# In[175]:


# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("Size of training set:", X_train.shape)
print("Size of test set:", X_test.shape)


# ### Random Forest (RF)

# In[176]:


#Training Text Classification Model and Predicting Sentiment by using RF
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 


# In[177]:


y_predRF = classifier.predict(X_test)
y_predRF


# In[178]:


#Model Evaluation for RF
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_predRF))
print(classification_report(y_test,y_predRF))
print(accuracy_score(y_test, y_predRF))


# In[179]:


# Calculating the accuracy of RF
import sklearn.metrics as metrics

acc_rf = round( metrics.accuracy_score(y_test, y_predRF) * 100 , 2 )
print( 'Accuracy of Random Forest model : ', acc_rf )


# In[180]:


#Saving the RF model
with open('text_classifierRF', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)


# In[181]:


#Testing the trained model
with open('text_classifierRF', 'rb') as training_model:
    model = pickle.load(training_model)


# In[182]:


y_predRF2 = model.predict(X_test)

print(confusion_matrix(y_test, y_predRF2))
print(classification_report(y_test, y_predRF2))
print(accuracy_score(y_test, y_predRF2)) 


# ### Gaussian Naive Bayes (GNB)

# In[183]:


#Training Text Classification Model and Predicting Sentiment by using GNB
from sklearn.naive_bayes import GaussianNB
import pickle  

gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[184]:


y_predGNB = gnb.predict(X_test)
y_predGNB


# In[185]:


#Model Evaluation for GNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_predGNB))
print(classification_report(y_test,y_predGNB))
print(accuracy_score(y_test, y_predGNB))


# In[186]:


# Calculating the accuracy for GNB model
acc_gnb = round( metrics.accuracy_score(y_test, y_predGNB) * 100 , 2 )
print( 'Accuracy of Gaussian Naive Bayes model : ', acc_gnb )


# In[187]:


#Saving the GNB model 
with open('text_classifierGNB', 'wb') as picklefile:
    pickle.dump(gnb,picklefile)


# In[188]:


#Testing the trained model
with open('text_classifierGNB', 'rb') as training_model:
    model = pickle.load(training_model)


# In[189]:


y_predGNB2 = model.predict(X_test)

print(confusion_matrix(y_test, y_predGNB2))
print(classification_report(y_test, y_predGNB2))
print(accuracy_score(y_test, y_predGNB2)) 


# ### Support Vector Machine (SVM)

# In[190]:


#Training Text Classification Model and Predicting Sentiment by using SVM
from sklearn.preprocessing import StandardScaler
# Creating scaled set to be used in SVM model to improve the results
sc = StandardScaler()
X_trainSvm = sc.fit_transform(X_train)
X_testSvm = sc.transform(X_test)


# In[191]:


# Import Library of Support Vector Machine model
from sklearn import svm

svc = svm.SVC()   #create a Support Vector Classifier
svc.fit(X_trainSvm,y_train)   #train the model using the training sets 

# Prediction on test data
y_predSVM = svc.predict(X_testSvm)
y_predSVM


# In[192]:


#Model Evaluation for SVM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_predSVM))
print(classification_report(y_test,y_predSVM))
print(accuracy_score(y_test, y_predSVM))


# In[193]:


# Calculating the accuracy for SVM model
acc_svm = round( metrics.accuracy_score(y_test, y_predSVM) * 100, 2 )
print( 'Accuracy of SVM model : ', acc_svm )


# In[194]:


#Saving the SVM model 
with open('text_classifierSVM', 'wb') as picklefile:
    pickle.dump(svc,picklefile)


# In[195]:


#Testing the trained model
with open('text_classifierSVM', 'rb') as training_model:
    model = pickle.load(training_model)


# In[196]:


y_predSVM2 = model.predict(X_testSvm)

print(confusion_matrix(y_test, y_predSVM2))
print(classification_report(y_test, y_predSVM2))
print(accuracy_score(y_test, y_predSVM2)) 


# In[197]:


#Comparing the models
models = pd.DataFrame({
    'Model': ['Random Forest', 'Gaussian Naive Bayes', 'Support Vector Machine'],
    'Score': [acc_rf, acc_gnb, acc_svm]})
models


#     Based on the results above, we can assure that Support Vector Machine (SVM) has the highest accuracy score among all the models in classifying the text (in this case; the tweets), followed by Random Forest (RF) and lastly Gaussian Naive Bayes (GNB) where the accuracy for SVM is 90.78%. All the trained models are saved for future used and we also have done the testing for the saved models on the same test set. From the testing, we have figured out that the accuracy scores are same like the first prediction testing.It is enough to justify because we are using the same dataset.
