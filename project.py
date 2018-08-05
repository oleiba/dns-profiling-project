# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:25:25 2018

@author: Yaniv Agman and Oded Leiba
"""

from __future__ import division
import os # to set working directory
import csv # to read/write csv files
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from IPython.core.display import display
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os.path
import math
import time
cwd = os.getcwd()
print(cwd)

DIR = "C:\\Users\\Administrator\\Documents\\MyResearch\\attack-methods\\dns-profiling-project"
# DIR = "/home/yaniv/src/dns-profiling-project"
DATA_DIR = 'DNS'
os.chdir(DIR)

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
    #actual_end_time = minimums['frame.time_relative']
    # print('actual_end_time = ' + str(actual_end_time))
    end_index = minimums['frame.number']
    # print('end_index = ', str(end_index))
    # print('actual_end_time - start_time = ', (actual_end_time - start_time))
    return end_index

# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = clf.predict(X_test)

    test_time = time.time() - t0

    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    pred_proba = clf.predict_proba(X_test)
    y_pred = pred_proba[:,1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    print("auc:   %0.3f" % auc)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    #print("confusion matrix:")
    #print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, auc, train_time, test_time, fpr, tpr


# #############################################################################
# Extract relevant data
    
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
        print('loop took ' + str(end_time - start_time) + ' s')
        df4 = df3[(df3['frame.number'] >= start_index) & (df3['frame.number'] <= max_index)]
        all_users_segments[idx].append(df4)
        corpus.append(' '.join(df4['dns.qry.name'].values))
        print('user #' + str(idx) + ': ' + str(len(all_users_segments[idx])) + ' records')

print('Total segments in corpus: ' + str(len(corpus)))

# #############################################################################
# Get features in bag of words representation

# Count occurrences (array of frequencies)
vectorizer = CountVectorizer(token_pattern="(?u)\\b[\\w.-]+\\b")
X = vectorizer.fit_transform(corpus)

# TF-IDF transformer
print(str(len(vectorizer.get_feature_names())) + ' features (different domain names)')
transformer = TfidfTransformer(smooth_idf=False)
X_TFIDF = transformer.fit_transform(X.toarray())

# #############################################################################
# For each user, train classifiers

accuracy_score_totals = []
auc_totals = []
training_time_totals = []
test_time_totals = []
for cur_user in range(NUM_OF_USERS):
    print('Training classifiers for user ' + str(cur_user))
    start_seg_idx = 0
    for i in range(cur_user):
        start_seg_idx = start_seg_idx + len(all_users_segments[i])
    end_seg_idx = start_seg_idx + len(all_users_segments[cur_user]) - 1
    print('User segments are ' + str(start_seg_idx) + ' to ' + str(end_seg_idx))
    
    # Set target according to user segments - 1 for a segment of the user, -1 otherwise
    user_target = np.empty(len(corpus))
    user_target.fill(-1)
    user_target[range(start_seg_idx,end_seg_idx+1)] = 1
    
    # Do undersampling as data is imbalanced (1 against 14)
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X_TFIDF, user_target)
    
    # Split corpus to X_train, X_test, and Y_train, Y_test
    # Todo: use k-fold cross validation instead, as in:
    # http://scikit-learn.org/stable/modules/cross_validation.html
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4)

    results = []
 
    print('=' * 80)
    print("kNN")
    clf1 = KNeighborsClassifier(n_neighbors=10)
    results.append(benchmark(clf1))
    
    print('=' * 80)
    print("Random forest")
    clf2 = RandomForestClassifier(n_estimators=100)
    results.append(benchmark(clf2))
    
    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    clf3 = MultinomialNB(alpha=.01)
    results.append(benchmark(clf3))
    clf4 = BernoulliNB(alpha=.01)
    results.append(benchmark(clf4))
    
    print('=' * 80)
    print("Voting ensemble")
    eclf = VotingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('mnb', clf3), ('bnb', clf4)], voting='soft')
    results.append(benchmark(eclf))


    # make some plots
    
    indices = np.arange(len(results))
    
    results = [[x[i] for x in results] for i in range(7)]
    
    clf_names, score, auc, training_time, test_time, fpr, tpr = results
    training_time_norm = np.array(training_time) / np.max(training_time)
    test_time_norm = np.array(test_time) / np.max(test_time)
    
    accuracy_score_totals.append(score)
    auc_totals.append(auc)
    training_time_totals.append(training_time)
    test_time_totals.append(test_time)
    
    # Plot all ROC curves
    plt.figure()
    lw = 2
    colors = itertools.cycle(['deeppink', 'navy', 'aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (area = {1:0.3f})'
                 ''.format(clf_names[i], auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of user {0}'.format(cur_user))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('.//Results//ROC' + str(cur_user) + '.png')


def calc_avgs(totals):
    totals_arr = [0, 0, 0, 0, 0]
    for i in range(len(totals)):
        for j in range(5):
            totals_arr[j] += totals[i][j]
    for j in range(5):
        totals_arr[j] = totals_arr[j] / len(totals)
    return totals_arr

accuracy_score_avgs =  calc_avgs(accuracy_score_totals)
auc_avgs = calc_avgs(auc_totals)
training_time_avgs = calc_avgs(training_time_totals)
test_time_avgs = calc_avgs(test_time_totals)
training_time_avg_norm = np.array(training_time_avgs) / np.max(training_time_avgs)
test_time_avg_norm = np.array(test_time_avgs) / np.max(test_time_avgs)

# Plot extra data as bars
fig, ax = plt.subplots(figsize=(12, 8)) 
# plt.title("Score")
ax.barh(indices, accuracy_score_avgs, .2, label="Accuracy", color='navy')
ax.barh(indices + .2, auc_avgs, .2, label="AUC", color='deeppink')
ax.barh(indices + .4, training_time_avg_norm, .2, label="Training time", color='c')
ax.barh(indices + .6, test_time_avg_norm, .2, label="Test time", color='darkorange')
ax.set_yticks(indices + 0.3)
ax.set_yticklabels(clf_names, minor=False)
for i, v in enumerate(accuracy_score_avgs):
    ax.text(v + 0.01, i, '{0:0.3f}'.format(v), color='black', fontweight='bold')
for i, v in enumerate(auc_avgs):
    ax.text(v + 0.01, i + .2, '{0:0.3f}'.format(v), color='black', fontweight='bold')
for i, v in enumerate(training_time_avgs):
    ax.text(training_time_avg_norm[i] + 0.01, i + .4, '{0:0.3f} sec'.format(v), color='black', fontweight='bold')
for i, v in enumerate(test_time_avgs):
    ax.text(test_time_avg_norm[i] + 0.01, i + .6, '{0:0.3f} sec'.format(v), color='black', fontweight='bold')
plt.legend(loc='lower left')
x0, x1, y0, y1 = plt.axis()
plt.axis((x0, x1 + 0.06, y0, y1))

# plt.show()
plt.savefig('.//Results//averages.png')
#############################################################################
