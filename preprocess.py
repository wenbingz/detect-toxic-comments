# coding=utf-8
import pandas as pd
import codecs
import pickle as pic
import nltk
import string
from nltk.corpus import stopwords as sw



class PreProcessor:

    def __init__(self, original_data_path, generate_data_path):
        self.__generate_data_path = generate_data_path
        self.__original_data_path = original_data_path

    def transform_data(self, auto_save=True, test = False):
        #columns = ["id","target","comment_text","severe_toxicity","obscene", "identity_attack","insult","threat","asian","atheist","bisexual","black","buddhist","christian","female","heterosexual","hindu","homosexual_gay_or_lesbian","intellectual_or_learning_disability","jewish","latino","male","muslim","other_disability","other_gender","other_race_or_ethnicity","other_religion","other_sexual_orientation","physical_disability","psychiatric_or_mental_illness","transgender","white","created_date","publication_id","parent_id","article_id","rating","funny","wow","sad","likes","disagree","sexual_explicit","identity_annotator_count","toxicity_annotator_count"]
        columns = ["id", "target", "comment_text"]
        if test:
            columns = ["id", "comment_text"]
        data = pd.read_csv(filepath_or_buffer=self.__original_data_path, sep=",", encoding="utf-8")
        print(data.columns.values.tolist())
        #data = data[['id', 'target', 'comment_text']]
        data = data[columns]
        #print(data)
        data = data.fillna('')
        data = data.drop_duplicates().reset_index(drop=True)
        if not test:
            comments_content = data[['id', 'target', 'comment_text']]
        else:
            comments_content = data[['id', 'comment_text']]
        if auto_save:
            comments_content.to_csv(self.__generate_data_path + "comments.csv", index=False, encoding='utf-8')

        return comments_content
    def generate_tokens(self, comment_content, auto_save=True, test = False):

        def _tokenization(text):
            cacheStopWords = sw.words("english")
            lowers = text.lower()  # remove the punctuation using the character deletion step of translate
            remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
            no_punctuation = lowers.translate(remove_punctuation_map)
            tokens = nltk.word_tokenize(no_punctuation)
            #print(cacheStopWords)
            return [w for w in tokens if w not in cacheStopWords]

        short_ids = []
        useful_ids = []
        targets = []
        res = []
        threshold = 2
        for i in range(comment_content.shape[0]):
            id = comment_content.loc[i, 'id']
            content = comment_content.loc[i, 'comment_text']
            if not test:
                target = comment_content.loc[i, 'target']
            #print("------", id)
            #print(content)
            result = _tokenization(content)
            #print("here")
            #print(result)
            if len(result) < threshold and not test:
                short_ids.append(id)
            else:
                useful_ids.append(id)
                res.append(result)
                if not test:
                    targets.append(target)
        if not test:
            comments_id_tokens = pd.DataFrame({'id': useful_ids, 'tokens': res, 'target': targets})
        else:
            comments_id_tokens = pd.DataFrame({'id': useful_ids, 'tokens':res})
        if auto_save:
            comments_id_tokens.to_csv(self.__generate_data_path + 'comments_id_tokens.csv', index=False, encoding='utf-8')
            f = open(self.__generate_data_path + 'short_ids.pdata', 'wb')
            pic.dump(short_ids, f)
            f.close()
        #return [comments_id_tokens, short_ids]

