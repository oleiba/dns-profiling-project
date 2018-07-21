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

DIR = "C:\\Users\\Administrator\\Documents\\MyResearch\\attack-methods\\project"
DATA_DIR = 'DNS'
DOMAIN_SUFFIXES_FILE_NAME = 'tld-desc.csv'

os.chdir(DIR)

with open(DOMAIN_SUFFIXES_FILE_NAME, 'r') as results:
    domainSuffixesReader = csv.reader(results, delimiter=',')
    domainSuffixes = list(domainSuffixesReader)

with open('dnsSummary_user292.pcap.csv', 'r') as results:
    masqueradeReader = csv.reader(results, delimiter=',')
    masqueradeRows = list(masqueradeReader)

NUM_OF_USERS = len(os.listdir('.\\' + DATA_DIR))
print(str(NUM_OF_USERS) + ' users')
for file in os.listdir('.\\' + DATA_DIR):
    

    
