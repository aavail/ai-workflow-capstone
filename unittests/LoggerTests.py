#!/usr/bin/env python
"""
model tests
"""

import os,sys
import unittest
## import model specific functions and variables
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from model_logging import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_trainlog(self):
        """
        test the train functionality
        """

        ## train the model
        model_train()
        model_predict()
        self.assertTrue(os.path.exists(TRAIN_LOG))
        self.assertTrue(os.path.exists(PREDICT_LOG))

  
### Run the tests
if __name__ == '__main__':
    unittest.main()
