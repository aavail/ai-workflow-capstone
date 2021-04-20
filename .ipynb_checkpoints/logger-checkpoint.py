#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','x_shape','eval_test','model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), data_shape, eval_test,
                            MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)

def update_predict_log(y_pred, y_proba, query, runtime, MODEL_VERSION, test=False):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','y_proba','query','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(), time.time(), y_pred, y_proba,query,
                            MODEL_VERSION, runtime])
        writer.writerow(to_write)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE
    
    ## train logger
    update_train_log(str((100,10)),"{'rmse':0.5}","00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
    ## predict logger
    update_predict_log("[0]", "[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
                       "00:00:01", MODEL_VERSION, test=True)
    
        
