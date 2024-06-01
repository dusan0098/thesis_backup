import dataiku
import pandas as pd
import numpy as np
import re
import json
import os

# Library Functions for Preprocessing Steps

# Remove urls from string columns
def remove_urls(df, columns):
    url_pattern = r'https?://\S+|www\.\S+'
    for column in columns:
        df[column] = df[column].apply(lambda text: re.sub(url_pattern, '', text) if isinstance(text, str) else text)
    return df

# Replace mentions with known real name or @user for unknown users
def replace_mentions(df, columns, username_to_name):
    for column in columns:
        df[column] = df[column].apply(lambda tweet: ' '.join([username_to_name.get(word[1:], '@user') if word.startswith('@') else word for word in tweet.split()]) if isinstance(tweet, str) else tweet)
    return df

# Remove hashtags from tweets - All of just those at the end of a tweet
def remove_hashtags(df, columns, on_end_only=False):
    if on_end_only:
        hashtag_pattern_end = r'\s(?:#\S+)$'
    else:
        hashtag_pattern_end = r'#\S+'
    
    for column in columns:
        if on_end_only:
            df[column] = df[column].apply(lambda tweet: re.sub(hashtag_pattern_end, '', tweet) if isinstance(tweet, str) else tweet)
        else:
            df[column] = df[column].apply(lambda tweet: re.sub(hashtag_pattern_end, '', tweet) if isinstance(tweet, str) else tweet)
    return df

# Remove tweets that don't have at least a certain number of words - in order to avoid biasing the clusters
def remove_short_tweets(df, columns, min_word_count):
    """
    Removes rows from the DataFrame where one or more of the specified columns
    have fewer than `min_word_count` words.

    Parameters:
    - df: pandas DataFrame containing the data.
    - columns: List of column names to check for word count.
    - min_word_count: Minimum number of words required for a tweet to be kept.

    Returns:
    - DataFrame with rows removed where any of the specified columns have fewer than `min_word_count` words.
    """

    def tweet_has_enough_words(tweet, min_count):
        # Split tweet into words, accounting for multiple spaces as single delimiter
        words = re.split(r'\s+', tweet.strip()) if isinstance(tweet, str) else []
        return len(words) >= min_count

    # Apply the filter for each specified column and keep rows where all checks pass
    for column in columns:
        df = df[df[column].apply(lambda tweet: tweet_has_enough_words(tweet, min_word_count))]

    return df

# Remove empty rows from dataframe - no text to work with
def remove_empty_rows(df, columns):
    for column in columns:
        df = df[df[column].apply(lambda x: isinstance(x, str) and x.strip() != '')]
    return df
