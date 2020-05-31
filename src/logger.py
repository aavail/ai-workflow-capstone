#!/usr/bin/env python
"""
logging a machine learning model to enable performance monitoring
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series using random forest"
LOG_DIR_PATH = os.path.join(os.path.dirname(__file__),'..','log')

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version = MODEL_VERSION, test=False, prefix='example'):
    """
    example function to update predict log file
    """
    
    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    logfile = os.path.join(LOG_DIR_PATH, "{}-predict-{}-{}.log".format(prefix, today.year, today.month))
    
    ## write the date to a csv file
    header = ['unique_id', 'timestamp', 'country', 'y_pred', 'y_proba', 'target_date', 'model_version', 'runtime', 'mode']
    write_header = False

    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        mode = test and 'test' or 'prod'
        to_write = map(str, [uuid.uuid4(), time.time(), country, y_pred, y_proba, target_date, model_version, runtime, mode])
        writer.writerow(to_write)

def update_train_log(country, date_range, metric, runtime, model_version = MODEL_VERSION, model_version_note = MODEL_VERSION_NOTE,
                     test=False, prefix='example'):
    """
    example function to update train log file
    """
    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    logfile = os.path.join(LOG_DIR_PATH, "{}-train-{}-{}.log".format(prefix, today.year, today.month))
    
    ## write the date to a csv file
    header = ['unique_id', 'timestamp', 'country', 'date_range', 'metric', 'model_version', 'model_version_note', 'runtime', 'mode']
    write_header = False

    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        mode = test and 'test' or 'prod'
        to_write = map(str, [uuid.uuid4(), time.time(), country, date_range, metric, model_version, model_version_note, runtime, mode])
        writer.writerow(to_write)
