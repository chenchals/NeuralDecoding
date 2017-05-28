import random, json

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils.knn import KNN
from config.global_config import PIPELINE_CONFIG

def run_model(session, verbose=False, debug=False):
    # names = ['DecisionTree']
    # names = ['NearestNeighbors', 'DecisionTree', "RandomForest", "AdaBoost"]
    hyper_parameter = PIPELINE_CONFIG.HYPER_PARAMETER
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

    selected = PIPELINE_CONFIG.MODEL_NAME

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
    hyper_parameter = PIPELINE_CONFIG.HYPER_PARAMETER
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


    selected = PIPELINE_CONFIG.MODEL_NAME
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