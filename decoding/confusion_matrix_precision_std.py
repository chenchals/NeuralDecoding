import cProfile
import io
import os
import pstats
from collections import Counter
from multiprocessing import Pool

from numpy import std

from decoding.pipeline import run_model, run_model_100
from utils.data_handler import load_data, get_time

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..')
data_folder = os.path.join(project_root, 'data')

# plot
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
'''
hypothesis:
The data processing is very slow, try improve the data processing part
'''

global alg_name

def preprocessing_load_test():
    return load_data()



def end_to_end_verbose(sessions, scalar=0.1, single_run=True):
    # each session is one file?
    length = scalar
    num_files = int(len(sessions) * length)
    print(num_files, end=',')
    p = Pool(8)
    sessions = sessions[:num_files]
    # return two things, aggregate and time series
    if single_run:
        ret = p.map(run_model, sessions)
    else:
        ret = p.map(run_model_100, sessions)
    p.close()
    p.join()
    # this is the time series
    if single_run:
        ts = [Counter(x[1]) for x in ret]
        count = len(ret)
    else:
        raise Exception()
    ts_all = {}
    # print(ts)
    for one_ts in ts:
        for time_point_ in one_ts:
            if time_point_ in ts_all:
                ts_all[time_point_].append(one_ts[time_point_])
            else:
                ts_all[time_point_] = [one_ts[time_point_]]
    # print(ts_all)
    # print([len(x) for x in ts_all.values()])
    ts_all_df = pd.DataFrame(ts_all)
    mat = ts_all_df.as_matrix()
    plt = sns.tsplot(data=mat)
    plt.set(xlabel='timepoint', ylabel='precision')
    sns.plt.title(alg_name)
    sns.plt.ylim(0,1)
    # sns.plt.show()
    sns.plt.savefig(os.path.join(project_root,'plots','{0}_{1}'.format(alg_name, str(num_files))))
    sns.plt.clf()
    # this is the aggregate
    aggregate = [Counter(x[0]) for x in ret]
    results = [list(x.values())[0] for x in aggregate]
    # compute std of precision
    print(std(results), end=',')
    all = Counter({})
    for each in aggregate:
        all = all + each

    # print(all)
    for alg in all:
        # print(alg, 'precision:', all[alg]/count)
        print(alg, all[alg]/count, end=',')
        # print(all[alg]/count,end=',')

        # for df in load_data():
    #     run_model(df)


def run_experiment(mode='Single'):
    for i in ["NearestNeighbors", "RBF SVM",
              "DecisionTree", "RandomForest", "AdaBoost", "KNN", "RadiusNeighbors"]:

        alg_name = i
        pr = cProfile.Profile()
        pr.enable()
        # ... do something ...
        sessions = preprocessing_load_test()

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        # print(s.getvalue())
        ps.print_stats()
        print('Read', len(sessions), 'files', 'in', get_time(s.getvalue()), 'secs')

        print('num_files,std,alg_name,avg_precision,time')
        for i in range(1, 2):
            pr.clear()
            pr.enable()
            # ... do something ...
            if mode.lower() == 'single':
                end_to_end_verbose(sessions, 0.05)
            else:
                run_model_100(sessions, 0.05)
            # end_to_end_verbose(sessions, 1*(i/20))
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            # print(s.getvalue())
            ps.print_stats()
            print(get_time(s.getvalue()))


# print()
# exit()
