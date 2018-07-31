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
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import NearestCentroid
import seaborn as sns # for heatmaps
import os, os.path
import math
import time
cwd = os.getcwd()
print(cwd)

# DIR = "C:\\Users\\Administrator\\Documents\\MyResearch\\attack-methods\\dns-profiling-project"
DIR = "/home/yaniv/src/dns-profiling-project"
DATA_DIR = 'DNS'
DOMAIN_SUFFIXES_FILE_NAME = 'tld-desc.csv'
PROCESSED_DIR = 'processed'
os.chdir(DIR)

with open(DOMAIN_SUFFIXES_FILE_NAME, 'r') as results:
    domainSuffixesReader = csv.reader(results, delimiter=',')
    domainSuffixes = list(domainSuffixesReader)

# NUM_OF_USERS = len(os.listdir('.\\' + DATA_DIR))
NUM_OF_USERS = len(os.listdir(DATA_DIR))
print(str(NUM_OF_USERS) + ' users')

# input: start index of DataFrame
# output: end index of 30 minutes segment, exclusive. Nan if that's the last segment
def find_segment_end(dataframe, start_index):
    start_time = dataframe[dataframe['frame.number'] == start_index]['frame.time_relative'].iloc[0]
    # print('start_time = ' + str(start_time))
    end_time = start_time + 60 * 30     # 60 seconds * 30 minutes
    # print('end_time = ' + str(end_time))
    data = dataframe[dataframe['frame.time_relative'] >= end_time]
    if data.empty:
        return np.nan
    minimums = data.iloc[0]
    #minimums = dataframe[dataframe['frame.time_relative'] >= end_time].min(axis=0)
    #actual_end_time = minimums['frame.time_relative']
    # print('actual_end_time = ' + str(actual_end_time))
    end_index = minimums['frame.number']
    # print('end_index = ', str(end_index))
    # print('actual_end_time - start_time = ', (actual_end_time - start_time))
    return end_index
    
all_users_segments = [[] for _ in range(NUM_OF_USERS)] # array of all users dataframes with all segments of each (sum of (user * segments_per_user))
corpus = [] # array of strings:  each string is a space separated array of domain names
# for idx, file in enumerate(os.listdir('.\\' + DATA_DIR)):
for idx, file in enumerate(os.listdir(DATA_DIR)):
    user_segments = [] # array of all segments of a single user. Each segment is of half an hour
    print(file)
    with open(os.path.join(DATA_DIR, file), 'r') as results:
        df = pd.read_csv(results, delimiter=',')
        print(str(len(df['dns.qry.name'].values)) + ' values pre-filter')
    
        # filter to have only IPv4 valid responses
        df2 = df[(df['dns.qry.type'] == 1) & (df['dns.flags.response'] == 1) & (df['dns.flags.rcode'] == 0)]
        # we take only the fields we need
        df2 = df2[['frame.number', 'frame.time_relative', 'dns.qry.name']]
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
        start_time = time.time()
        while not math.isnan(end_index):
            df4 = df3[(df3['frame.number'] >= start_index) & (df3['frame.number'] < end_index)]
            #print(df4)
            all_users_segments[idx].append(df4)
            corpus.append(' '.join(df4['dns.qry.name'].values))
            start_index = end_index
            end_index = find_segment_end(df3, start_index)
        # last segment
        end_time = time.time()
        print('loop took ' + str(end_time - start_time) + ' ms')
        df4 = df3[(df3['frame.number'] >= start_index) & (df3['frame.number'] <= max_index)]
        all_users_segments[idx].append(df4)
        corpus.append(' '.join(df4['dns.qry.name'].values))
        print('user #' + str(idx) + ': ' + str(len(all_users_segments[idx])) + ' records')

print(len(corpus))
# For each user: Need to split corpus to X_train and X_test and create Y_train and Y_test
# Y_train and Y_test should be easy to create- set 1 for a segment of the user, set -1 otherwise
# Train should contain samples of both the user and not of the user, better if balanced
# Then, we should fit_transform on X_train only, and not on all corpus
# See example in:
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py

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


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    #print("classification report:")
    #print(metrics.classification_report(y_test, pred,
    #                                    target_names=target_names))

    #print("confusion matrix:")
    #print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
#############################################################################


################## Why do we need this??

# collect averaged weights for ALL users together
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(20)
len(weights_df)

# create TD-IDF features for each user
users_data = {}
for feature_name in vectorizer.get_feature_names():
    users_data[feature_name] = []
# for user_index, file in enumerate(os.listdir('.\\' + DATA_DIR)):
for user_index, file in enumerate(os.listdir(DATA_DIR)):
    print(file)
    for feature_index, feature_name in enumerate(vectorizer.get_feature_names()):
        users_data[feature_name].append(transformed_weights[user_index, feature_index])
users_df = pd.DataFrame(data=users_data)
processed_path = os.path.join(PROCESSED_DIR, 'users.csv')
users_df.to_csv(path_or_buf=processed_path)


