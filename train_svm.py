import numpy as np
import pandas as pd
from sklearn import svm
from ast import literal_eval
from sklearn.externals import joblib

class Svm_Builder:
    def __init__(self, id_tokenv_path, model_path):
        self.id_tokenv_path = id_tokenv_path
        self.model_path = model_path
    def generate(self):
        vectors = []
        labels = []

        id_tokenv = pd.read_csv(self.id_tokenv_path, encoding='utf-8')
        counter = 200000
        for i in range(id_tokenv.shape[0]):
            v = id_tokenv.loc[i, 'tokens']
            target = id_tokenv.loc[i, 'target']
            v = literal_eval(v)
            if target > 0.001:
                labels.append(1)
                vectors.append(v[0])
                counter -= 1
                if counter == 0:
                    break
        print("amount of non-zero: ", counter)
        counter = 200000
        for i in range(id_tokenv.shape[0]):
            if i % 10000 == 0:
                print(str(i) + " of " + str(id_tokenv.shape[0]))
            v = id_tokenv.loc[i, 'tokens']
            target = id_tokenv.loc[i, 'target']
            v = literal_eval(v)
            target = float(target)
            if target <= 0.001:
                labels.append(0)
                vectors.append(v[0])
                counter -= 1
                if counter == 0:
                    break

        print("length of vector length: ", len(vectors))

        print('starting to configure parameters')

        clf = svm.SVC(kernel='rbf', class_weight='balanced', verbose = True)
        print('starting to fit dataset')
        clf.fit(vectors, labels)
        print('starting to output')
        with open(self.model_path + "svm.model", "wb") as fw:
            joblib.dump(clf, fw)
