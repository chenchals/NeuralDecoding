from utils.data_handler import get_file_list, get_sessions
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import os, random, cProfile
from collections import Counter
from utils.knn import KNN
from numpy import std
import cProfile, pstats, io
import numpy as np

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



def end_to_end_verbose(sessions, scalar=0.1):
    # each session is one file?
    length = scalar
    num_files = int(len(sessions) * length)
    print(num_files, end=',')
    p = Pool(8)
    sessions = sessions[:num_files]
    # return two things, aggregate and time series
    ret = p.map(run_model, sessions)
    p.close()
    p.join()
    # this is the time series
    ts = [Counter(x[1]) for x in ret]
    count = len(ret)
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

def load_data():
    # each session is a file, each file has 49 time points
    data_set = get_sessions(get_file_list(data_folder))
    return data_set


def run_model(session, verbose=False, debug=False):
    # names = ['DecisionTree']
    # names = ['NearestNeighbors', 'DecisionTree', "RandomForest", "AdaBoost"]
    hyper_parameter = 1
    if verbose:
        print(hyper_parameter, end=', ')
    names = ["NearestNeighbors", "RBF SVM",
             "DecisionTree", "RandomForest", "AdaBoost", "KNN", "RadiusNeighbors"]
    classifiers = [
        KNeighborsClassifier(hyper_parameter),
        SVC(gamma=hyper_parameter, C=1),
        DecisionTreeClassifier(max_depth=hyper_parameter),
        RandomForestClassifier(max_depth=hyper_parameter, n_estimators=10, max_features=1),
        AdaBoostClassifier(n_estimators=hyper_parameter),
        KNN(num_neighbors=hyper_parameter, weight='distance'),
        RadiusNeighborsClassifier(hyper_parameter)]

    classifier_pool = dict(zip(names, classifiers))

    # selected = ["KNN"]
    selected = [alg_name] # "AdaBoost"
    # selected = ["NearestNeighbors", "RBF SVM", "DecisionTree", "RandomForest", "AdaBoost", "KNN", "RadiusNeighbors"]
    results = dict(zip(selected, [0 for _ in range(len(selected))]))
    counter = 0
    # setup
    # names = ['NearestNeighbors']
    classifiers = [classifier_pool[k] for k in selected]
    # print(len(sessions))
    time_points = dict()
    for time_point in session:
        timestamp = 0
        time_point_data = time_point[1]
        inputs = []
        labels = []
        counter += 1
        for record in time_point_data.iterrows():
            dat = record[1]
            if dat['timestamp'] not in time_points:
                timestamp = int(dat['timestamp'])
                time_points[timestamp] = 0
            inputs.append([dat['f1'], dat['f2']])
            labels.append(int(dat['label']-1))

        if debug:
            # to make sure each time point data has the time point value
            assert len(list(time_points.keys())) == 1
        # print(time_set)
        take_one_out = random.randint(0, len(inputs)-1)
        # print(len(labels), take_one_out)
        X_test = [inputs[take_one_out]]
        y_test = [labels[take_one_out]]
        # X_train = inputs[1:]
        # y_train = labels[1:]
        X_train = inputs[:take_one_out] + inputs[take_one_out:]
        y_train = labels[:take_one_out] + labels[take_one_out:]
        # print(each)
        # iterate over classifiers
        for name, clf in zip(selected, classifiers):
            # print(name, clf)
            clf.fit(X_train, y_train)
            my_answer = clf.predict(X_test)
            score = accuracy_score(my_answer, y_test)
            results[name] += score
            time_points[timestamp] = score
    # print(time_points)
    for key in results:
        results[key] /= counter
    if verbose:
        print(results)
    return results, time_points
    # print(results)


def run_model_100(session, verbose=False, debug=False):
    # names = ['DecisionTree']
    # names = ['NearestNeighbors', 'DecisionTree', "RandomForest", "AdaBoost"]
    hyper_parameter = 1
    if verbose:
        print(hyper_parameter, end=', ')
    names = ["NearestNeighbors", "RBF SVM",
             "DecisionTree", "RandomForest", "AdaBoost", "KNN", "RadiusNeighbors"]
    classifiers = [
        KNeighborsClassifier(hyper_parameter),
        SVC(gamma=hyper_parameter, C=1),
        DecisionTreeClassifier(max_depth=hyper_parameter),
        RandomForestClassifier(max_depth=hyper_parameter, n_estimators=10, max_features=1),
        AdaBoostClassifier(n_estimators=hyper_parameter),
        KNN(num_neighbors=hyper_parameter, weight='distance'),
        RadiusNeighborsClassifier(hyper_parameter)]

    classifier_pool = dict(zip(names, classifiers))

    # selected = ["KNN"]
    selected = [alg_name] # "AdaBoost"
    # selected = ["NearestNeighbors", "RBF SVM", "DecisionTree", "RandomForest", "AdaBoost", "KNN", "RadiusNeighbors"]
    results = dict(zip(selected, [0 for _ in range(len(selected))]))
    counter = 0
    # setup
    # names = ['NearestNeighbors']
    classifiers = [classifier_pool[k] for k in selected]
    # print(len(sessions))
    time_points = dict()
    for time_point in session:
        timestamp = 0
        time_point_data = time_point[1]
        inputs = []
        labels = []
        scores = []
        counter += 1
        for record in time_point_data.iterrows():
            dat = record[1]
            if dat['timestamp'] not in time_points:
                timestamp = int(dat['timestamp'])
                # initialize time_points
                time_points[timestamp] = None
            inputs.append([dat['f1'], dat['f2']])
            labels.append(int(dat['label']-1))

        if debug:
            # to make sure each time point data has the time point value
            assert len(list(time_points.keys())) == 1
        # print(time_set)
        confusion_matrix = []

        for name, clf in zip(selected, classifiers):
            # heavy computation here, run 100 times per time point
            for _ in range(100):
                take_one_out = random.randint(0, len(inputs)-1)
                # print(len(labels), take_one_out)
                X_test = [inputs[take_one_out]]
                y_test = [labels[take_one_out]]
                X_train = inputs[:take_one_out] + inputs[take_one_out:]
                y_train = labels[:take_one_out] + labels[take_one_out:]
                # print(name, clf)
                clf.fit(X_train, y_train)
                my_answer = clf.predict(X_test)
                # I need to record confusion matrix here
                # compare y_test with my_answer to calculate that
                # 11, 10, 00, 01
                score = accuracy_score(my_answer, y_test)
                confusion_matrix.append(''.join([str(my_answer), str(y_test)]))
                # record 100 calls for avg and std
                scores.append(score)
                time_points[timestamp] = confusion_matrix
            results[name] = scores
    # print(time_points)
    for key in results:
        results[key] /= counter
    if verbose:
        print(results)
    return results, time_points












def get_time(string):
    return string.split('\n')[0].strip().split(' ')[-2]






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
    for i in range(1,2):
        pr.clear()
        pr.enable()
        # ... do something ...
        end_to_end_verbose(sessions, 0.05)
        # end_to_end_verbose(sessions, 1*(i/20))
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        # print(s.getvalue())
        ps.print_stats()
        print(get_time(s.getvalue()))


# print()
# exit()
