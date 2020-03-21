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
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(DATA_PATH,'unittest',False,'germany')
        trained_model = path.join(MODEL_PATH,'unittest-germany-0_1.joblib')
        self.assertTrue(os.path.exists(trained_model))
#        os.remove(trained_model)
        

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## load the model
        _,models = model_load('germany')
#        print(models)
        model = list(models.values())[0]
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    def test_03_predict(self):
        """
        test the predict functionality
        """

        ## load model first
        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day)
        ## example predict
        self.assertTrue(result is not None)

        
        

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
