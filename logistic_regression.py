import pandas as pd
from sklearn.linear_model import LogisticRegression
from ast import literal_eval
import pickle

class Logistic_Regression_Builder:
    def __init__(self, id_tokenv_path, model_path):
        self.id_tokenv_path = id_tokenv_path
        self.model_path = model_path
    def generate(self):
        vectors = []
        labels = []

        id_tokenv = pd.read_csv(self.id_tokenv_path, encoding='utf-8')
        for i in range(id_tokenv.shape[0]):
            v = id_tokenv.loc[i, 'tokens']
            target = id_tokenv.loc[i, 'target']
            v = literal_eval(v)
            target = float(target)
            if target >= 0.001:
                labels.append(0)
                vectors.append(v[0])
        lr_model = LogisticRegression()
        lr_model.fit(vectors, labels)
        with open(self.model_path + "logistic.model", "wb") as fw:
            pickle.dumps(lr_model, fw)

