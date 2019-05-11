from gensim import corpora, models
import pandas as pd
import pickle as pic
from ast import literal_eval


class TfIdfGenerator:

    def __init__(self, generate_data_path, original_tokens_path=''):
        self.__original_tokens_path = original_tokens_path
        self.__generate_data_path = generate_data_path
        self.__comments_tokens = None

    def generate(self, comments_tokens=None, auto_save=True):

        if comments_tokens is None:
            assert (self.__original_tokens_path != ''), 'No data source.'
            self.__comments_tokens = pd.read_csv(self.__original_tokens_path, encoding='utf-8')
        else:
            self.__comments_tokens = comments_tokens
        pure_tokens = self.__comments_tokens['tokens'].values.tolist()
        pure_tokens = [literal_eval(t) for t in pure_tokens]  # transform list-like strings to list
        print(pure_tokens)
        exit(-1)
        word_dict = corpora.Dictionary(pure_tokens)  # Used in LSA or LDA algo
        comments_bow = [word_dict.doc2bow(t) for t in pure_tokens]
        algo = models.TfidfModel(comments_bow)
        corpus_tfidf = algo[comments_bow]
        comments_vec = []
        for t in corpus_tfidf:
            comments_vec.append([v for (_, v) in t])
        newsid_tfvec = pd.DataFrame({'id': self.__comments_tokens['id'].values.tolist(), 'tf_vec': comments_vec, 'target': self.__comments_tokens['target'].values.tolist()})
        if auto_save:
            f = open(self.__generate_data_path + 'corpus_tfidf.pdata', 'wb')
            pic.dump(corpus_tfidf, f)
            f.close()
            word_dict.save(self.__generate_data_path + 'word_dict.dict')
            newsid_tfvec.to_csv(self.__generate_data_path + 'comments_id_tfvec.csv', index=False, encoding='utf-8')
        return newsid_tfvec
