#!/usr/bin/env python3

"""
collection of functions for the final case study solution
"""

import os
import pandas as pd

def load_json_data(data_dir) :
    """
    laod all json formatted files into a dataframe
    """
    
    ## Define the directory containing the JSON files
    #directory = 'cs-train'
    
    ## Define the column names of the ingested dataframe
    columns = ['country', 'customer_id', 'invoice', 'price', 'stream_id', 'times_viewed', 'year', 'month', 'day']
    
    ## Initialize an empty list to hold the DataFrames
    dataframes = []
    
    ## Loop through each file in the directory
    for file_name in os.listdir(data_dir):
        
        ## Check if the file is a JSON file
        if file_name.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(data_dir, file_name)
            
            # Read the JSON data into a DataFrame and append to the list
            splited_df = pd.read_json(file_path)
            splited_df = pd.DataFrame(splited_df.values)
            dataframes.append(splited_df)
    
    ## Merge all DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    ## Adding the clumns headers
    df.columns = columns
    
    ## Convert each column to its appropriate data types
    df['country'] = df['country'].astype('category')           # Convert to category for better memory usage
    df['customer_id'] = df['customer_id'].astype('Int64')      # Convert to nullable integer
    df['invoice'] = df['invoice'].astype(str)                  # Keep as string
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to float, coerce errors to NaN
    df['stream_id'] = df['stream_id'].astype(str)              # Keep as string
    df['times_viewed'] = df['times_viewed'].astype(int)        # Convert to integer
    df['year'] = df['year'].astype(int)                        # Convert to integer
    df['month'] = df['month'].astype(int)                      # Convert to integer
    df['day'] = df['day'].astype(int)                          # Convert to integer
    
    ## Adding a invoice_date column to the dataframe & convert it to the appropriate data type
    df['invoice_date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['invoice_date'] = df['invoice_date'].astype('datetime64[s]')
    
    ## Sort the DataFrame by the 'date' column where the most recent dates are at the top.
    df.sort_values(by='invoice_date', ascending=True, inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    return(df.head())

if __name__ == "__main__":

    # Specify the directory
    data_dir = 'cs-train'  
    dataframe = load_json_data(data_dir)
    
    # Initializing the output print
    print('fetching data ...')
    
    # Print the first few rows of the DataFrame
    print(dataframe.head())  