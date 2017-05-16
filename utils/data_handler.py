import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def get_file_list(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

def read_data(file_path):
    df = pd.read_csv(file_path, names=['timestamp', 'label', 'f1', 'f2'], dtype={'timestamp': np.int32, 'label': np.int32, 'f1': np.float32, 'f2': np.float64}, na_values=['n/a'])
    return df

def stats(df):
    df['label'] = df['label'].astype(np.int32)
    df['timestamp'] = df['timestamp'].astype(np.int32)
    grouped = df.groupby(['label']).size()
    # print(grouped)
    grouped = df.groupby(['timestamp']).size()
    print(grouped)
    for row in df[['timestamp', 'label', 'f1', 'f2']].iterrows():
        print(row[1])

def get_sessions(file_list):
    '''
    Each file has 49 time points of something (fixed number time points). Each time point has multiple reads from inputs.
    This reader divides a file into multiple chunks grouped by time points
    Each data frame df is data from one file grouped by time points (fixed number time points), each time points has various number of readings.
    :return:  A list of data frame generated from files in file_list
    '''
    sessions = []
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')


    for each_file in file_list:
        data_path = os.path.join(project_root, 'data', each_file)
        df = read_data(data_path).groupby(['timestamp'])
        sessions.append(df)
    return sessions

