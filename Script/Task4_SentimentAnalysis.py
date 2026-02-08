------------#Step 1: Import Libraries--------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob

-------------#Step 2: Load Dataset-------------
df = pd.read_csv(r"C:\Users\admin\Documents\Prodigy Infotech\Prodigy_DS_04\Datasets\Tweets\Tweets.csv")

df.head()

-------------#Step 3: Understand Data-----------
#Dataset Info
df.info()

#Check Missing Values
df.isnull().sum()

-------------#Step 4.1: Drop unnecessary columns-------------
df = df.drop(columns=[
    'negativereason', 
    'negativereason_confidence', 
    'airline_sentiment_gold',
    'negativereason_gold', 
    'tweet_coord', 
    'tweet_location', 
    'user_timezone'
])

#Step 4.2: Confirm missing values
df.isnull().sum()

-------------#Step 5: Select Text Column-------------
df[['text', 'airline_sentiment']].head()

-------------#Step 6: Sentiment Visualization---------------
#Count sentiment distribution
sns.countplot(x='airline_sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

-------------#Step 7: Sentiment by Airline----------------
sns.countplot(x='airline', hue='airline_sentiment', data=df)
plt.xticks(rotation=45)
plt.title("Sentiment by Airline")
plt.show()

------------#Step 8: Text Length Analysis---------------
df['text_length'] = df['text'].apply(len)

sns.histplot(df['text_length'], bins=30)
plt.title("Tweet Length Distribution")
plt.show()

-----------#Step 9: Polarity Analysis using TextBlob------------
# Polarity: -1 (negative) to 1 (positive)
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

plt.figure(figsize=(8,5))
sns.histplot(df['polarity'], bins=30, kde=True, color='lightgreen')
plt.title("Sentiment Polarity Distribution")
plt.xlabel("Polarity Score")
plt.ylabel("Number of Tweets")
plt.show()


------------#INSIGHTS-----------------
print("✅ Task 4 Insights:\n")

# Total tweets
print(f"Total Tweets analyzed: {len(df)}")

# Sentiment counts
print("\nTweet Sentiment Counts:")
print(df['airline_sentiment'].value_counts())

# Average polarity
print(f"\nAverage Tweet Polarity: {df['polarity'].mean():.2f}")

# Airline with most negative tweets
negative_counts = df[df['airline_sentiment']=='negative']['airline'].value_counts()
print(f"\nAirline with most negative tweets: {negative_counts.idxmax()} ({negative_counts.max()} tweets)")



------------#Sentiment Distribution Visualization----------------
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment distribution bar chart
plt.figure(figsize=(6,4))
sns.countplot(x='airline_sentiment', data=df, hue='airline_sentiment', dodge=False, palette={'negative':'red', 'neutral':'gray', 'positive':'green'}, legend=False)
plt.title('Tweet Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()


# Pie chart for sentiment distribution
sentiment_counts = df['airline_sentiment'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['red', 'gray', 'green'])
plt.title('Sentiment Proportion')
plt.show()








