#!/usr/bin/env python
"""
model tests
"""
import unittest
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "src"))
## import model specific functions and variables
from model import *

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cs-train')

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """
        ## train the model
        model_train(prefix='ut', data_dir=DATA_DIR, test=False, countries=['united_kingdom'])
        SAVED_MODEL = os.path.join(MODEL_DIR, "ut-united_kingdom-0_1.joblib")
        self.assertTrue(os.path.exists(SAVED_MODEL))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        _, models = model_load(prefix='ut', data_dir=DATA_DIR, countries=['united_kingdom'])
        model = list(models.values())[0]
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    def test_03_predict(self):
        """
        test the predict function input
        """

        country='united_kingdom'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day, prefix='ut')
        ## ensure that a list can be passed

        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] is not None)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
