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
    sessions = []
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')
    for each_file in file_list:
        data_path = os.path.join(project_root, 'data', each_file)
        df = get_timepoints_data(data_path)
        sessions.append(df)
    return sessions

def get_timepoints_data(file_path):
    df = read_data(file_path)
    # df['label'] = df['label'].astype(np.int32)
    # df['timestamp'] = df['timestamp'].astype(np.int32)
    return df.groupby(['timestamp'])

if __name__ == "__main__":
    import os
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')
    data_path = os.path.join(project_root, 'data', 'iTYPE_1_iCE_0007_iCL_2_mi_af10_168_01_SP009a_wf_ALL02_REWTIM_01')
    df = read_data(data_path)
    # stats(df)
    get_timepoints_data(data_path)