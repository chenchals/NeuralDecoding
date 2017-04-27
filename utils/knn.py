from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator

class KNN(object):
    # weight supports uniform and distance
    def __init__(self, num_neighbors = 10, weight='uniform'):
        self.num_neighbors = num_neighbors
        self.dict = {}
        self.weight = weight

    def fit(self, x, y):
        '''
        fit function to fit the model
        :param x: list of inputs - numpy.ndarray() shape=(num_samples, num_features)
        :param y: list of labels - numpy.ndarray() shape=(num_samples, 1)
        :return:
        '''
        self.x = x
        self.y = y

    def predict(self, x, num_neighbors=None):
        '''
        predict function to predict labels
        :param x: list of inputs - numpy.ndarray() shape=(num_samples, num_features)
        :return: list of labels - numpy.ndarray() shape=(num_samples, 1)
        '''
        ret = []
        for each in x:
            sim = cosine_similarity(self.x, np.array(each).reshape(1, -1))
            if num_neighbors is None:
                num_neighbors = self.num_neighbors
            arg = np.argsort(sim, axis=0)[::-1][:num_neighbors][0]
            # print(arg, len(sim))
            tmp = {}
            for i in arg:
                if self.y[i] in tmp:
                    if self.weight == 'uniform':
                        tmp[self.y[i]] += 1
                    else:
                        tmp[self.y[i]] += sim[i]
                else:
                    if self.weight == 'uniform':
                        tmp[self.y[i]] = 1
                    else:
                        tmp[self.y[i]] = sim[i]
            ret.append(max(tmp.items(), key=operator.itemgetter(1))[0])
        return ret
