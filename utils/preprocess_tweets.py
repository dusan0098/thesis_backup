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

def replace_mentions(df, columns, username_to_name, replace_fully=False):
    """
    Replace or remove mentions in the specified columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - columns: List of column names to process.
    - username_to_name: Dictionary mapping usernames to real names.
    - replace_fully: If True, completely remove mentions from the text.

    Returns:
    - DataFrame with mentions processed as specified.
    """
    mention_pattern = r'@\w+'

    for column in columns:
        if replace_fully:
            print("Removing mentions completely")
            df[column] = df[column].apply(lambda tweet: re.sub(mention_pattern, '', tweet) if isinstance(tweet, str) else tweet)
        else:
            print("Substituting mentions")
            df[column] = df[column].apply(lambda tweet: ' '.join([username_to_name.get(word[1:], '@user') if word.startswith('@') else word for word in tweet.split()]) if isinstance(tweet, str) else tweet)
    
    return df

# Remove hashtags from tweets - All of just those at the end of a tweet
def remove_hashtags(df, columns, on_end_only=False, remove_fully=True):
    """
    Remove hashtags from tweets in specified columns.

    Parameters:
    - df: pandas DataFrame containing the data.
    - columns: List of column names to process.
    - on_end_only: If True, only remove hashtags at the end of the tweet.
    - remove_fully: If True, remove the entire hashtag. If False, remove only the '#' symbol.

    Returns:
    - DataFrame with hashtags processed as specified.
    """
    if on_end_only:
        if remove_fully:
            hashtag_pattern = r'\s#\S+$'  # Matches a space followed by # and any non-space characters until the end of the tweet.
        else:
            hashtag_pattern = r'#(\S+)$'  # Captures the word following the # symbol at the end of the tweet.
    else:
        if remove_fully:
            hashtag_pattern = r'#\S+'  # Matches # and any non-space characters.
        else:
            hashtag_pattern = r'#(\S+)'  # Captures the word following the # symbol.

    for column in columns:
        if on_end_only:
            if remove_fully:
                print("Removing hashtags at end of tweet - fully")
                df[column] = df[column].apply(lambda tweet: re.sub(hashtag_pattern, '', tweet) if isinstance(tweet, str) else tweet)
            else:
                print("Removing hashtags at end of tweet - # symbol only")
                df[column] = df[column].apply(lambda tweet: re.sub(hashtag_pattern, r'\1', tweet) if isinstance(tweet, str) else tweet)
        else:
            if remove_fully:
                print("Removing all hashtags - fully")
                df[column] = df[column].apply(lambda tweet: re.sub(hashtag_pattern, '', tweet) if isinstance(tweet, str) else tweet)
            else:
                print("Removing all hashtags - # symbol only")
                df[column] = df[column].apply(lambda tweet: re.sub(hashtag_pattern, r'\1', tweet) if isinstance(tweet, str) else tweet)
    
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

# Replace truncated tweets with the retweet text
def replace_truncated_text(df, text_column, retweet_column):
    def replace_text(row):
        full_text = row[text_column]
        retweet_full_text = row[retweet_column]
        
        if (full_text.endswith('...') or full_text.endswith('…')) and pd.notna(retweet_full_text) and retweet_full_text.strip() != '':
            return retweet_full_text
        return full_text
    
    df[text_column] = df.apply(replace_text, axis=1)
    return df

def replace_entities(text, entity_dict):
    for entity, char in entity_dict.items():
        text = text.replace(entity, char)
    return text

# Remove html entitites
def remove_html_entities(df, columns):
    html_entities = ['&amp;', '&lt;', '&gt;', '&quot;', '&#39;']
    
    for column in columns:
        for entity in html_entities:
            df[column] = df[column].str.replace(entity, '', regex=False)
    
    return df



# Replace German umlauts
def replace_german_umlauts(df, columns):
    umlauts = {
        'Ä': 'Ae',
        'ä': 'ae',
        'Ö': 'Oe',
        'ö': 'oe',
        'Ü': 'Ue',
        'ü': 'ue',
        'ß': 'ss',
    }
    for column in columns:
        df[column] = df[column].apply(lambda text: replace_entities(text, umlauts) if isinstance(text, str) else text)
    return df
