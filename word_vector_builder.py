from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from ast import literal_eval
import pandas as pd

class Word2Vector_Builder:
    def __init__(self, generate_data_path, original_tokens_path=''):
        self.__original_tokens_path = original_tokens_path
        self.__generate_data_path = generate_data_path
        self.__comments_tokens = None
    def bulid_word2vector(self, comments_tokens=None, auto_save=True):
        if comments_tokens is None:
            assert (self.__original_tokens_path != ''), 'No data source.'
            self.__comments_tokens = pd.read_csv(self.__original_tokens_path, encoding='utf-8')
        else:
            self.__comments_tokens = comments_tokens
        pure_tokens = self.__comments_tokens['tokens'].values.tolist()
        pure_tokens = [literal_eval(t) for t in pure_tokens]  # transform list-like strings to list
        model = Word2Vec(pure_tokens, size=20, window=5, min_count=1, workers=4)
        model.save("word2vec_wb.model")
        #model = Word2Vec.load("word2vec.model")
        print("over--------------")
    def libs(self):
        path = get_tmpfile("word2vec.model")
        print(path)
        print(common_texts)
        model = Word2Vec(common_texts, size=6, window=5, min_count=1, workers=4)

        model.save("word2vec.model")
        model = Word2Vec.load("word2vec.model")
        vector = model.wv['computer']  # numpy vector of a word
        print('#'*100)
        print(vector)