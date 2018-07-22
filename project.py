# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:25:25 2018

@author: Administrator
"""

from __future__ import division
import os # to set working directory
import csv # to read/write csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn.preprocessing import Imputer # for imputing missing values
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns # for heatmaps
import os, os.path
import math
import time
cwd = os.getcwd()
print(cwd)

DIR = "C:\\Users\\Administrator\\Documents\\MyResearch\\attack-methods\\dns-profiling-project"
DATA_DIR = 'DNS'
DOMAIN_SUFFIXES_FILE_NAME = 'tld-desc.csv'
PROCESSED_DIR = 'processed'
os.chdir(DIR)

with open(DOMAIN_SUFFIXES_FILE_NAME, 'r') as results:
    domainSuffixesReader = csv.reader(results, delimiter=',')
    domainSuffixes = list(domainSuffixesReader)

NUM_OF_USERS = len(os.listdir('.\\' + DATA_DIR))
print(str(NUM_OF_USERS) + ' users')

# input: start index of DataFrame
# output: end index of 30 minutes segment, exclusive. Nan if that's the last segment
def find_segment_end(dataframe, start_index):
    start_time = dataframe[dataframe['frame.number'] == start_index]['frame.time_relative'].iloc[0]
    # print('start_time = ' + str(start_time))
    end_time = start_time + 60 * 30     # 60 seconds * 30 minutes
    # print('end_time = ' + str(end_time))
    minimums = dataframe[dataframe['frame.time_relative'] >= end_time].min(axis=0)
    actual_end_time = minimums['frame.time_relative']
    # print('actual_end_time = ' + str(actual_end_time))
    end_index = minimums['frame.number']
    # print('end_index = ', str(end_index))
    # print('actual_end_time - start_time = ', (actual_end_time - start_time))
    return end_index
    
all_users_segments = [] # array of all users dataframes with all segments of each (sum of (user * segments_per_user))
corpus = [] # array of strings:  each string is a space separated array of domain names
users_num_of_segments = [] # array of integers: each element is the number of segments per user
for idx, file in enumerate(os.listdir('.\\' + DATA_DIR)):
    user_segments = [] # array of all segments of a single user. Each segment is of half an hour
    print(file)
    with open(os.path.join(DATA_DIR, file), 'r') as results:
        df = pd.read_csv(results, delimiter=',')
        print(str(len(df['dns.qry.name'].values)) + ' values pre-filter')
    
        # filter to have only IPv4 valid responses
        df2 = df[(df['dns.qry.type'] == 1) & (df['dns.flags.response'] == 1) & (df['dns.flags.rcode'] == 0)]
        df2.index = range(len(df2))
        print(str(len(df2['dns.qry.name'].values)) + ' values post-filter')
        
        # special cases 
        df3 = df2.copy()
        df3['dns.qry.name'] = df2['dns.qry.name'].replace(value='whatsapp.net', regex='.*whatsapp.*', inplace=False)
        print(str(len(df3['dns.qry.name'].values)) + ' post group by whatsapp')
        print(len(df3['dns.qry.name'].values))
    
        # slice into segments
        start_index = df3.min(axis=0)['frame.number']
        end_index = find_segment_end(df3, start_index)
        max_index = df3.max(axis=0)['frame.number']
        count = 0
        start_time = time.time()
        while not math.isnan(end_index):
            # print('#'  + str(count))
            count += 1
            df4 = df3[(df3['frame.number'] >= start_index) & (df3['frame.number'] < end_index)]
            all_users_segments.append(df4)
            all_names_as_text = ' '.join(df3['dns.qry.name'].values)
            corpus.append(all_names_as_text)
            start_index = end_index
            end_index = find_segment_end(df3, start_index)
            end_time = time.time()
        # last segment
        end_time = time.time()
        print('loop took ' + str(end_time - start_time) + ' ms')
        df4 = df3[(df3['frame.number'] >= start_index) & (df3['frame.number'] <= max_index)]
        all_users_segments.append(df4)
        all_names_as_text = ' '.join(df3['dns.qry.name'].values)
        corpus.append(all_names_as_text)
        users_num_of_segments.append(count + 1)
        print('user #' + str(idx) + ': ' + str(count + 1) + ' records')

# Count occurrences (array of frequencies)
vectorizer = CountVectorizer(token_pattern="(?u)\\b[\\w.-]+\\b")
X = vectorizer.fit_transform(corpus)

# collect summed frequencies for ALL users together
frequencies = np.asarray(X.sum(axis=0)).ravel().tolist()
frequencies_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'frequency': frequencies})
frequencies_df.sort_values(by='frequency', ascending=False).head(20)

# TF-IDF transformer
print(str(len(vectorizer.get_feature_names())) + ' features (different domain names)')
transformer = TfidfTransformer(smooth_idf=False)
transformed_weights = transformer.fit_transform(X.toarray())

# collect averaged weights for ALL users together
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(20)
len(weights_df)

# create TD-IDF features for each user
users_data = {}
for feature_name in vectorizer.get_feature_names():
    users_data[feature_name] = []
for user_index, file in enumerate(os.listdir('.\\' + DATA_DIR)):
    print(file)
    for feature_index, feature_name in enumerate(vectorizer.get_feature_names()):
        users_data[feature_name].append(transformed_weights[user_index, feature_index])
users_df = pd.DataFrame(data=users_data)
processed_path = os.path.join(PROCESSED_DIR, 'users.csv')
users_df.to_csv(path_or_buf=processed_path)


