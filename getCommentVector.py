from gensim.models import Word2Vec
import pandas as pd
from ast import literal_eval
import numpy as np

class comment_vector_builder:
    def __init__(self, w2v_model_path, token_list_path, v_size):
        self.w2v_model_path = w2v_model_path
        self.token_list_path = token_list_path
        self.v_size = v_size

    def generate(self, generate_data_path):
        id_comments_tokens = pd.read_csv(self.token_list_path, encoding='utf-8')
        model = Word2Vec.load(self.w2v_model_path)
        counter = 0
        comments_v = []
        ids = []
        targets = []
        for i in range(id_comments_tokens.shape[0]):
            comment_v = np.zeros(self.v_size).reshape((1, self.v_size))
            id = id_comments_tokens.loc[i, 'id']
            tokens = id_comments_tokens.loc[i, 'tokens']
            target = id_comments_tokens.loc[i, 'target']
            tokens = literal_eval(tokens)
            counter = 0
            for t in tokens:
                comment_v += np.array(model.wv[t]).reshape((1, self.v_size))

                counter += 1
            comment_v /= counter
            comments_v.append(comment_v.tolist())
            ids.append(id)
            targets.append(target)
        id_token = pd.DataFrame({"id": ids, "tokens":comments_v, "target": targets})
        id_token.to_csv(generate_data_path + "id_tokenv.csv", index=False, encoding='utf-8')






