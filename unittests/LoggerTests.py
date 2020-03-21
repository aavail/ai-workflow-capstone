#!/usr/bin/env python
"""
model tests
"""


import unittest
## import model specific functions and variables
import os,sys
from os import path
wdir = path.join(path.dirname(__file__),'..')
sys.path.append(wdir)
from src.model import model_train,model_load,model_predict
MODEL_PATH = path.join(wdir,'models')
DATA_PATH = path.join(wdir,'cs-train')
LOG_PATH = path.join(wdir,'log')

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train_log(self):
        """
        test the train log functionality
        """

        ## train the model
        model_train(DATA_PATH,'unittest',False,'germany')
        self.assertTrue(path.exists(path.join(LOG_PATH,'example-predict-2020-3.log')))
#        os.remove(trained_model)
        

    def test_02_predict_log(self):
        """
        test the predict log functionality
        """

        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day)
        ## example predict
        log_path = path.join(LOG_PATH,'example-training-2020-3.log')
        self.assertTrue(path.exists(log_path))


        
### Run the tests
if __name__ == '__main__':
    unittest.main()
