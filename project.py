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
cwd = os.getcwd()
print(cwd)

DIR = "C:\\Users\\Administrator\\Documents\\MyResearch\\attack-methods\\dns-profiling-project"
DATA_DIR = 'DNS'
DOMAIN_SUFFIXES_FILE_NAME = 'tld-desc.csv'
os.chdir(DIR)

with open(DOMAIN_SUFFIXES_FILE_NAME, 'r') as results:
    domainSuffixesReader = csv.reader(results, delimiter=',')
    domainSuffixes = list(domainSuffixesReader)

NUM_OF_USERS = len(os.listdir('.\\' + DATA_DIR))
print(str(NUM_OF_USERS) + ' users')
# for file in os.listdir('.\\' + DATA_DIR):
file = 'dnsSummary_user292.pcap.csv'
print(file)
with open(os.path.join(DATA_DIR, file), 'r') as results:
    df = pd.read_csv(results, delimiter=',')
    df.head()
    print(df['dns.qry.name'].values)
    print(str(len(df['dns.qry.name'].values)) + ' values pre-filter')
    
    # filter to have only IPv4 queries
    dfOnlyQueries = df[(df['dns.qry.type'] == 1) & (df['dns.flags.response'] == 0)]
    print(dfOnlyQueries['dns.qry.name'].values)
    print(str(len(dfOnlyQueries['dns.qry.name'].values)) + ' values post-filter')
    print(len(dfOnlyQueries['dns.qry.name'].values))
    
    # count occurrences
    dfOnlyQueries['dns.qry.name'].value_counts()
