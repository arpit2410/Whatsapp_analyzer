import pandas as pd
import re
import emoji
def preprocess(data):
    pattern = r'([0-9]+\/[0-9]+\/[0-9]+,\s[0-9]+:[0-9]+(?:\s(?:AM|PM|am|pm))?\s-\s)'
    msg_and_dates = re.split(pattern, data)
    msg_and_dates = [part.strip() for part in msg_and_dates if part.strip()]
    msgs = msg_and_dates[1::2]
    dates = msg_and_dates[::2]
    df = pd.DataFrame({'user_messages': msgs, 'dates': dates})
    df['dates'] = pd.to_datetime(df['dates'], format='%d/%m/%Y, %H:%M -')
    df.rename(columns={'dates': 'date'}, inplace=True)
    users = []
    messages = []
    for message in df['user_messages']:
        s = message
        s = s.split(":")
        if len(s) == 2:
            spiltmessage = message.split(':')
            author = spiltmessage[0]
            m = ' '.join(spiltmessage[1:])
            users.append(author)
            messages.append(m)
        else:
            users.append('group_notification')
            messages.append(message)

    df['user'] = users
    df['messages'] = messages
    df.drop(columns=['user_messages'], inplace=True)

    def extract_emojis(text):
        emoji_pattern = re.compile("[\U00010000-\U0010ffff]")  # Unicode emoji range
        return emoji_pattern.findall(text)

    df['emojis'] = df['messages'].apply(extract_emojis)
    year = df['date'].dt.year
    month = df['date'].dt.month_name()
    dates = df['date'].dt.day
    hour = df['date'].dt.hour
    minute = df['date'].dt.minute
    df['day'] = dates
    df['month'] = month
    df['year'] = year
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = hour
    df['minute'] = minute
    df['Month_Num'] = df['date'].dt.month
    df['only_date'] = df['date'].dt.date
    return df