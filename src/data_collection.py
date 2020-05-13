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
from collections import defaultdict
#logging = logging.getlogging(__name__)
logging.basicConfig(level=logging.INFO)
import time
import warnings
warnings.filterwarnings("ignore")

def fetch_data(path,sample = True):
    """fetch json data from a path """
    correct_cols_name =sorted(['country', 'customer_id', 'day', 'invoice', 'month', 'price',
       'stream_id', 'times_viewed', 'year'])
    all_json_data = []
    logging.info(f'start loading data...')
    for i in glob.glob(os.path.join(path,'*.json')):
        with open(i) as f:
            data = json.load(f)
   
        unstandardized_element = []
        for idx,item in enumerate(data):
            if 'total_price' in item.keys():
                data[idx]['price'] = data[idx].pop('total_price')
                logging.debug(f'key_name total_price  is not standardized and has been changed to price in file {i}')
            if 'StreamID' in item.keys():
                data[idx]['stream_id'] = data[idx].pop('StreamID')
            if 'TimesViewed' in item.keys():
                data[idx]['times_viewed'] = data[idx].pop('TimesViewed')
            if  sorted(list(item.keys())) != correct_cols_name:
                logging.warning(f"key name of element {idx} in file {i} is {item.keys().__str__()}" +
                                "and it has been removed from file")
                unstandardized_element.append(idx)
        data = list( data[i] for i in range(len(data)) if i not in unstandardized_element)
        all_json_data.extend(data)
        if sample:break
    return pd.DataFrame(all_json_data)

def convert_to_ts(df_all,country ='United Kingdom'): 
    logging.info('start aggragating data')
    try:     
        df= df_all[df_all.country==country] 
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']],format='%Y%m%d')
        df_agg = df.groupby('date',as_index=False)[['price','times_viewed']].agg({'price':['sum','count'],'times_viewed':'sum'})
        df_agg.columns = ['date','revenue','purchases','total_views']
        df_agg['unique_streams'] = df.groupby('date',as_index=False)['stream_id'].transform(lambda x:x.nunique())
        df_agg['unique_invoices'] = df.groupby('date',as_index=False)['invoice'].transform(lambda x:x.nunique())
    #    fill empty date
        df_agg = df_agg.set_index(df_agg['date'])
        df_agg.index = pd.DatetimeIndex(df_agg.index)
        idx = pd.date_range(df_agg.index.min(), df_agg.index.max())
        df_agg = df_agg.reindex(idx, fill_value=0)
    except Exception as e:
        logging.exception("agg failed",exc_info=True)
    return df_agg


def engineer_features(df,training=True):
    """
    for any given day the target becomes the sum of the next days revenue
    for that day we engineer several features that help predict the summed revenue
    
    the 'training' flag will trim data that should not be used for training
    when set to false all data will be returned

    """

    ## extract dates
    logging.info('start feature')
    dates = df['date'].values.copy()
    dates = dates.astype('datetime64[D]')

    ## engineer some features
    eng_features = defaultdict(list)
    previous =[7, 14, 28, 70]  #[7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    y = np.zeros(dates.size)
    for d,day in enumerate(dates):

        ## use windows in time back from a specific date
        for num in previous:
            current = np.datetime64(day, 'D') 
            prev = current - np.timedelta64(num, 'D')
            mask = np.in1d(dates, np.arange(prev,current,dtype='datetime64[D]'))
            eng_features["previous_{}".format(num)].append(df[mask]['revenue'].sum())

        ## get get the target revenue    
        plus_30 = current + np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(current,plus_30,dtype='datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        ## attempt to capture monthly trend with previous years data (if present)
        start_date = current - np.timedelta64(365,'D')
        stop_date = plus_30 - np.timedelta64(365,'D')
        mask = np.in1d(dates, np.arange(start_date,stop_date,dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        ## add some non-revenue features
        minus_30 = current - np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(minus_30,current,dtype='datetime64[D]'))
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean())
        eng_features['recent_views'].append(df[mask]['total_views'].mean())

    X = pd.DataFrame(eng_features)
    ## combine features in to df and remove rows with all zeros
#    X.fillna(0,inplace=True)
#    mask = X.sum(axis=1)>0
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    X.reset_index(drop=True, inplace=True)

#    if training == True:
#        ## remove the last 30 days (because the target is not reliable)
#        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
#        X = X[mask]
#        y = y[mask]
#        dates = dates[mask]
#        X.reset_index(drop=True, inplace=True)
    
    return(X,y,dates)

def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds) 

if __name__ == "__main__":

    run_start = time.time() 
    path = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(path,'..',"cs-train"))
    print(data_dir)
    data_all = fetch_data(data_dir,False)
    logging.info("load time: {}".format(convert(time.time()-run_start)))
    aggr_df = convert_to_ts(data_all)
    logging.info("aggr time: {}".format(convert(time.time()-run_start)))
    featured_df = engineer_features(aggr_df,False)
    logging.info("feature time: {}".format(convert(time.time()-run_start)))





