from utils.data_handler import get_timepoints_data, get_file_list
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import os, random, cProfile

names = ["Nearest Neighbors", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost"]
names = [['Nearest Neighbors', 'Decision Tree', "Random Forest", "AdaBoost"][0:0]]

'''
hypothesis:
The data processing is very slow, try improve the data processing part
'''
def single_run():
    names = ['Decision Tree']
    # names = ['Nearest Neighbors', 'Decision Tree', "Random Forest", "AdaBoost"]
    print(names)
    hyper_parameter =1
    print(hyper_parameter, end=', ')
    classifiers = [
        KNeighborsClassifier(hyper_parameter),
        SVC(gamma=hyper_parameter, C=1),
        DecisionTreeClassifier(max_depth=hyper_parameter),
        RandomForestClassifier(max_depth=hyper_parameter, n_estimators=10, max_features=1),
        AdaBoostClassifier(n_estimators=hyper_parameter)]

    classifier_pool = dict(zip(names, classifiers))

    # setup
    # names = ['Nearest Neighbors']
    classifiers = [classifier_pool[k] for k in names]

    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')
    data_folder = os.path.join(project_root, 'data')
    results = dict(zip(names, [0 for _ in range(len(names))]))
    counter = 0
    data_set = []
    for each_file in get_file_list(data_folder):
        data_path = os.path.join(project_root, 'data', each_file)
        df = get_timepoints_data(data_path)
        data_set.append(df)
    for df in data_set:
        X_test = y_test = X_train = y_train = []

        for each in df:
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
            for name, clf in zip(names, classifiers):
                # print(name, clf)
                clf.fit(X_train, y_train)
                my_answer = clf.predict(X_test)
                score = accuracy_score(my_answer, y_test)
                results[name] += score
                # score = clf.score(my_answer, y_test)
                # print(name, score)
    for key in results:
        results[key] /= counter
        print(results[key])
        # print(results)
# single_run()
# exit()
cProfile.run('single_run()')
