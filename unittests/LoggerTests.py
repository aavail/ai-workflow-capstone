#!/usr/bin/env python
"""
model tests
"""
from datetime import date
import os
import sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "src"))
## import model specific functions and variables
from logger import update_train_log, update_predict_log

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'log')
LOG_PREFIX = 'unittests'

class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        ensure log file is created
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "{}-train-{}-{}.log".format(LOG_PREFIX, today.year, today.month))
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        country = 'india'
        date_range = ('2017-11-29', '2019-05-24')
        metric = {'rmse':0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"
        
        update_train_log(country, date_range, metric, runtime,
                         model_version, model_version_note, test=True, prefix=LOG_PREFIX)

        self.assertTrue(os.path.exists(log_file))
        
    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "{}-train-{}-{}.log".format(LOG_PREFIX, today.year, today.month))
        
        ## update the log
        country = 'india'
        date_range = ('2017-11-29', '2019-05-24')
        metric = {'rmse':0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"
        
        update_train_log(country, date_range, metric, runtime,
                         model_version, model_version_note ,test=True, prefix=LOG_PREFIX)

        df = pd.read_csv(log_file)
        logged_metric = [literal_eval(i) for i in df['metric'].copy()][-1]
        self.assertEqual(metric,logged_metric)
                

    def test_03_predict(self):
        """
        ensure log file is created
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "{}-predict-{}-{}.log".format(LOG_PREFIX, today.year, today.month))
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        y_pred = [0]
        y_proba = [0.6,0.4]
        runtime = "00:00:02"
        model_version = 0.1
        country = "india"
        target_date = '2018-01-05'

        update_predict_log(country, y_pred,y_proba,target_date,runtime,
                           model_version, test=True, prefix=LOG_PREFIX)
        
        self.assertTrue(os.path.exists(log_file))

    
    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "{}-predict-{}-{}.log".format(LOG_PREFIX, today.year, today.month))

        ## update the log
        y_pred = [0]
        y_proba = [0.6,0.4]
        runtime = "00:00:02"
        model_version = 0.1
        country = "india"
        target_date = '2018-01-05'

        update_predict_log(country, y_pred,y_proba,target_date,runtime,
                           model_version, test=True, prefix=LOG_PREFIX)

        df = pd.read_csv(log_file)
        logged_y_pred = [literal_eval(i) for i in df['y_pred'].copy()][-1]
        self.assertEqual(y_pred,logged_y_pred)


### Run the tests
if __name__ == '__main__':
    unittest.main()
      
