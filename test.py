# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:39:27 2020

@author: liu
"""
import os
wdir = 'D:/github/ai-workflow-capstone'
os.chdir(wdir)
from src.data_collection import *
logging.basicConfig(level=logging.INFO)


run_start = time.time() 
data_dir = "./cs-train"
os.path.abspath(data_dir)
data_all = fetch_data(data_dir,False)
logging.info("load time: {}".format(convert(time.time()-run_start)))
aggr_df = aggr_data(data_all)
logging.info("aggr time: {}".format(convert(time.time()-run_start)))
featured_df = engineer_features(aggr_df,False)
logging.info("feature time: {}".format(convert(time.time()-run_start)))
