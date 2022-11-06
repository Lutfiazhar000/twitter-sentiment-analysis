# import packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from textblob import TextBlob
from wordcloud import WordCloud

# query data from twitter 
query = "blue tick lang:en since:2022-11-01 -filter:links"
tweets = []
limit = 500

# looping for every tweet and save to df
try:
    print("start Scraping...")
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.content])

    df = pd.DataFrame(tweets, columns=['tweet'])

except:
    print("scraping failed")

# function for data cleaning
def clean_text(text):
    text = re.sub(r'@([#])|([^a-zA-Z])',' ',text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'RT[\s]+', ' ', text)

    return text

# cleaning data
df['tweet']= df['tweet'].apply(clean_text)

# function for get subjectivvity
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# function for get polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# create new column for subjectivity and polarity
df['Subjectivity'] = df['tweet'].apply(get_subjectivity)
df['Polarity'] = df['tweet'].apply(get_polarity)

# function for scoring negative, netral, and positive analysis
def get_analysis_score(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Netral'
    else:
        return 'Positive'

df['analysis'] = df['Polarity'].apply(get_analysis_score)

# print all positive tweets
j = 1
print('===POSITIVE TWEETS===')
sorted_positive_tweets = df.sort_values(by=['Polarity'])
for i in range(0, sorted_positive_tweets.shape[0]):
    if (sorted_positive_tweets['analysis'][i] == 'Positive'):
        print(str(i) + ') ' + sorted_positive_tweets['tweet'][i])
        print()
        j = j+1

# print all negative tweets
j = 1
print('===NEGATIVE TWEETS===')
sorted_negative_tweets = df.sort_values(by=['Polarity'], ascending = False)
for i in range(0, sorted_negative_tweets.shape[0]):
    if (sorted_negative_tweets['analysis'][i] == 'Negative'):
        print(str(j) + ') ' + sorted_negative_tweets['tweet'][i])
        print()
        j = j+1

# print all netral tweets
j = 1
print("===NETRAL TWEETS===")
sorted_netral_tweets = df.sort_values(by=['Polarity'], ascending = False)
for i in range(0, sorted_netral_tweets.shape[0]):
    if (sorted_netral_tweets['analysis'][i] == 'Netral'):
        print((str(j) + ') '+ sorted_netral_tweets['tweet'][i]))
        print()
        j = j+1

# get percentage of positive tweets
positive_tweets = df[df.analysis == 'Positive']
positive_tweets = positive_tweets['tweet']
percentage_positive_tweets = round((positive_tweets.shape[0] / df.shape[0] * 100), 1)
print('percentage of positive tweets = ', percentage_positive_tweets, "%")

# get percentage of negative tweets
negative_tweets = df[df.analysis == 'Negative']
negative_tweets = negative_tweets['tweet']
percentage_negative_tweets = round((negative_tweets.shape[0] / df.shape[0] * 100), 1)
print('percentage of negative tweets = ', percentage_negative_tweets, "%")

# get percentage of netral tweets
netral_tweets = df[df.analysis == 'Netral']
netral_tweets = netral_tweets['tweet']
percentage_netral_tweets = round((netral_tweets.shape[0] / df.shape[0] * 100), 1)
print('percentage of netral tweets = ', percentage_netral_tweets, "%")

# show the values counts
df['analysis'].value_counts()

# === display ===

# plot & visualize the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['analysis'].value_counts().plot(kind='bar', color='orange')
plt.show()

# plot polarity & subjectivity
plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='#EC7272')

plt.title("Sentiment Analysis")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.show()

# create wordcloud
all_word = ' '.join([tweets for tweets in df['tweet']])
wordcloud = WordCloud(width = 500, height = 300, random_state = 20, max_font_size = 200).generate(all_word)

plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()











