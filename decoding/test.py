from utils.data_handler import get_timepoints_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(1),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


if __name__ == "__main__":
    import os, random
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')
    data_path = os.path.join(project_root, 'data', 'iTYPE_1_iCE_0007_iCL_2_mi_af10_168_01_SP009a_wf_ALL02_REWTIM_01')
    df = get_timepoints_data(data_path)
    X_test = y_test = X_train = y_train = []
    results = dict(zip(names, [0 for _ in range(len(names))]))
    counter = 0
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
        X_test = [inputs[0]]
        y_test = [labels[0]]
        X_train = inputs[1:]
        y_train = labels[1:]
        # X_train = inputs[:take_one_out] + inputs[take_one_out:]
        # y_train = labels[:take_one_out] + labels[take_one_out:]
        # print(each)
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            my_answer = clf.predict(X_test)
            score = accuracy_score(my_answer, y_test)
            results[name] += score
            # score = clf.score(my_answer, y_test)
            # print(name, score)
    for key in results:
        results[key] /= counter
        print(key, results[key])
    # print(results)