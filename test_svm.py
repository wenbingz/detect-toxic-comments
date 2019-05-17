import numpy as np
import pandas as pd
from sklearn import svm
from ast import literal_eval
from sklearn.externals import joblib

class Svm_Test_Builder:
    def __init__(self, id_tokenv_path, model_path, test_generate_path):
        self.id_tokenv_path = id_tokenv_path
        self.model_path = model_path
        self.test_generate_path = test_generate_path
    def generate(self):
        vectors = []
        labels = []
        id_tokenv = pd.read_csv(self.id_tokenv_path, encoding='utf-8')
        ids = []
        tokens = []
        scores = []
        svm_m = joblib.load(self.model_path)
        for i in range(id_tokenv.shape[0]):
            if i % 100 == 0:
                print(i, ' ---- ', id_tokenv.shape[0])
            id_ = id_tokenv.loc[i, 'id']
            token = id_tokenv.loc[i, 'tokens']
            token = literal_eval(token)
            scores.append(svm_m.predict(token))
            ids.append(id_)
            tokens.append(token)
        id_token_score = pd.DataFrame({'id': ids, 'tokens': tokens, 'scores': scores})
        id_token_score.to_csv(self.test_generate_path + "id_token_score.csv", index = False, encoding = 'utf-8')

            
          

