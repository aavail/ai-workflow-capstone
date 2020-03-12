# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 23:12:09 2020

@author: liu

fetch data from json files
"""
import os
import re
import numpy as np
import pandas as pd
import json
import glob
import logging
logging.basicConfig(level=logging.info)
#prod_data = 'D:\\github\\ai-workflow-capstone\\cs-production'
#train_data = 'D:\\github\\ai-workflow-capstone\\cs-production\\'


def fetch_data(path,sample = True):
    """fetch json data from a path """
    correct_cols_name =sorted(['country', 'customer_id', 'day', 'invoice', 'month', 'price',
       'stream_id', 'times_viewed', 'year'])
    all_json_data = []
    for i in glob.glob(os.path.join(path,'*')):
        with open(i) as f:
            data = json.load(f)
   
        unstandardized_element = []
        for idx,item in enumerate(data):
            if 'total_price' in item.keys():
                data[idx]['price'] = data[idx].pop('total_price')
                logging.debug(f'key_name total_price  is not standardized,'+ 
                                ' and has been changed to price in file {i}')
            if 'StreamID' in item.keys():
                data[idx]['stream_id'] = data[idx].pop('StreamID')
            if 'TimesViewed' in item.keys():
                data[idx]['times_viewed'] = data[idx].pop('TimesViewed')
            if  sorted(list(item.keys())) != correct_cols_name:
                logging.warning(f'key name of element {idx} in file {i} is {item.keys().__str__()}' +
                                'and it has been removed from file')
                unstandardized_element.append(idx)
        data = list( data[i] for i in range(len(data)) if i not in unstandardized_element)
        all_json_data.extend(data)
        if sample:break
    return pd.DataFrame(all_json_data)

#all_keys = fetch_data('D:\\github\\ai-workflow-capstone\\test\\',False)
#if __name__
#all_keys = fetch_data(prod_data,False)