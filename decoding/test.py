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




'''
hypothesis:
The data processing is very slow, try improve the data processing part
'''

def preprocessing_load_test():
    return load_data()



def end_to_end_verbose(sessions, scalar=0.1):
    length = scalar
    num_files = int(len(sessions) * length)
    print(num_files, end=',')
    p = Pool(8)
    sessions = sessions[:num_files]
    ret = p.map(run_model, sessions)
    p.close()
    p.join()
    ret = [Counter(x) for x in ret]
    # compute std of precision
    results = [list(x.values())[0] for x in ret]
    print(std(results), end=',')
    count = len(ret)
    all = Counter({})
    for each in ret:
        all = all + each
    for alg in all:
        # print(alg, 'precision:', all[alg]/count)
        print(alg, all[alg]/count, end=',')
        # print(all[alg]/count,end=',')

        # for df in load_data():
    #     run_model(df)

def load_data():
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')
    data_folder = os.path.join(project_root, 'data')
    # each session is a file, each file has 49 time points
    data_set = get_sessions(get_file_list(data_folder))
    return data_set


def run_model(session, verbose=False):
    # names = ['Decision Tree']
    # names = ['Nearest Neighbors', 'Decision Tree', "Random Forest", "AdaBoost"]
    hyper_parameter = 1
    if verbose:
        print(hyper_parameter, end=', ')
    names = ["Nearest Neighbors", "RBF SVM",
             "Decision Tree", "Random Forest", "AdaBoost", "KNN", "RadiusNeighbors"]
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
    selected = ["AdaBoost"] # "AdaBoost"
    # selected = ["Nearest Neighbors", "RBF SVM", "Decision Tree", "Random Forest", "AdaBoost", "KNN", "RadiusNeighbors"]
    results = dict(zip(selected, [0 for _ in range(len(selected))]))
    counter = 0
    # setup
    # names = ['Nearest Neighbors']
    classifiers = [classifier_pool[k] for k in selected]

    for each in session:
        data = each[1]
        inputs = []
        labels = []
        counter += 1
        for record in data.iterrows():
            dat = record[1]
            inputs.append([dat['f1'], dat['f2']])
            labels.append(int(dat['label']-1))
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
    for key in results:
        results[key] /= counter
    if verbose:
        print(results)
    return results
        # print(results)

def get_time(string):
    return string.split('\n')[0].strip().split(' ')[-2]

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
    end_to_end_verbose(sessions, 1)
    # end_to_end_verbose(sessions, 1*(i/20))
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    # print(s.getvalue())
    ps.print_stats()
    print(get_time(s.getvalue()))


# print()
# exit()
