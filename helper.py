import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import emoji
def fetchmsg(user_type,df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    num_messages = df.shape[0]
    words = []
    for msg in df['messages']:
        words.extend(msg.split())
    media_count= df[df['messages'].str.contains('<Media omitted>')].shape[0]
    URLPATTERN = r'(https?://\S+)'
    df['urlcount'] = df.messages.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
    urlcount=df.loc[df['urlcount'] > 0].shape[0]
    emojilen = sum(df['emojis'].str.len())
    return num_messages,len(words),media_count,urlcount, emojilen

def most_busy_user(df):
    df = df[df['user'] != 'group_notification']
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df

def create_wordcloud(user_type,df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    wc= WordCloud(height=500, width=500, min_font_size=10, background_color='white')
    df_wc= wc.generate(df['messages'].str.cat(sep=" "))
    return df_wc

def most_common_word(user_type, df):
    f = open('stop_hinglish.txt', 'r', encoding='utf-8')
    stop_words = f.read()
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    temp = df[df['user'] != 'group_notification']
    words = []
    for msg in temp['messages']:
        for word in msg.lower().split():
            if word not in stop_words and word != '<media' and word != 'omitted>':
                words.append(word)
    most_df = pd.DataFrame(Counter(words).most_common(20))
    return most_df
def emojihelper(user_type, df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    emoji = []
    for msg in df['emojis']:
        emoji.extend(msg)
    femoji = pd.DataFrame(Counter(emoji).most_common(len(Counter(emoji))))
    return femoji
def timeline(user_type, df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    timeline =df.groupby(['year','Month_Num','month']).count()['messages'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline
def daily_timeline(user_type,df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    daily_timeline = df.groupby('only_date').count()['messages'].reset_index()
    return daily_timeline

def user_chat(user_type, df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    temp = df[df['user'] != 'group_notification']
    musers = pd.DataFrame(temp['user'].value_counts())
    musers = musers.reset_index()
    musers.columns = ["Users", "Counts"]
    return musers
def active_week(user_type, df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    mweek = pd.DataFrame(df['day_name'].value_counts())
    mweek = mweek.reset_index()
    mweek.columns = ["Day", "Counts"]
    return mweek
def active_month(user_type, df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    mmonth = pd.DataFrame(df['month'].value_counts())
    mmonth = mmonth.reset_index()
    mmonth.columns = ["Month", "Counts"]
    return mmonth
def findsent(data):
    if data["positive"] >= data["negative"] and data["positive"] >= data["neutral"]:
        return 1
    if data["negative"] >= data["positive"] and data["negative"] >= data["neutral"]:
        return -1
    if data["neutral"] >= data["positive"] and data["neutral"] >= data["negative"]:
        return 0
def sentiments(user_type, df):
    if user_type != 'Overall':
        df = df[df['user'] == user_type]
    data = df.dropna()
    sentiments = SentimentIntensityAnalyzer()
    data["positive"] = [sentiments.polarity_scores(message)["pos"] for message in data["messages"]]
    data["negative"] = [sentiments.polarity_scores(message)["neg"] for message in data["messages"]]
    data["neutral"] = [sentiments.polarity_scores(message)["neu"] for message in data["messages"]]
    data['value'] = data.apply(lambda row: findsent(row), axis=1)
    user_list = data['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")
    return data
def percentage(df,k):
    df = round((df['user'][df['value']==k].value_counts() / df[df['value']==k].shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df
# def most_senti(user_type, temp, k):
#     f = open('stop_hinglish.txt', 'r', encoding='utf-8')
#     stop_words = f.read()
#     words = []
#     for message in temp['messages'][temp['value'] == k]:
#         for word in message.lower().split():
#             if word not in stop_words and word != '<media' and word != 'omitted>' and not emoji.demojize(word):
#                 words.append(word)
#     most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])
#
#     return most_common_df
