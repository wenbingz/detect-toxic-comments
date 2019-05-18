import pandas as pd
from sklearn.linear_model import LogisticRegression
from ast import literal_eval
from sklearn.externals import joblib

class Logistic_Test_Builder:
    def __init__(self, id_token_score_path, model_path, test_generate_path):
        self.id_token_score_path = id_token_score_path
        self.model_path = model_path
        self.test_generate_path = test_generate_path
    def generate(self):
        ids_final = []
        values = []
        id_token_score = pd.read_csv(self.id_token_score_path, encoding='utf-8')
        logistic_m = joblib.load(self.model_path)
        for i in range(id_token_score.shape[0]):
            if i % 100 == 0:
                print(i, ' ---- ', id_token_score.shape[0])
            id_ = id_token_score.loc[i, 'id']
            token = id_token_score.loc[i, 'tokens']
            token = literal_eval(token)
            score = id_token_score.loc[i, 'scores']
            score = literal_eval(score)
            #print(score)
            if int(score[0]) == 1:
                values.append(logistic_m.predict_proba(token)[0][1])
            else:
                values.append(0)
            ids_final.append(id_)
        id_token_score = pd.DataFrame({'id': ids_final, 'target': values})
        id_token_score.to_csv(self.test_generate_path + "test_result.csv", index = False, encoding = 'utf-8')

