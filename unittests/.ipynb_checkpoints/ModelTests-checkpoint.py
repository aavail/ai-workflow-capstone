#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-0_1.joblib")))

    def test_02_load(self):
        """
        test the load functionality
        """
                        
        ## load the model
        data, models = model_load()
        
        self.assertTrue('predict' in dir(models['eire']))
        self.assertTrue('fit' in dir(models['eire']))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

       ## ensure that a list can be passed
        query = {'country': 'all', 'year': '2018', 'month': '01', 'day': '05', 'model': 'sl'}

        result = model_predict(query, test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] > 0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
