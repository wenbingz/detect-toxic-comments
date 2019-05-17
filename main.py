# coding=utf-8

from preprocess import PreProcessor
from tf_idf_build import TfIdfGenerator
from word_vector_builder import Word2Vector_Builder
from getCommentVector import comment_vector_builder
from train_svm import Svm_Builder
from test_svm import Svm_Test_Builder
from logistic_regression import Logistic_Regression_Builder
from logistic_test import Logistic_Test_Builder
import sys

original_data_path = './train/'
generate_data_path = './generate_datas/'
model_path = './models/'
test_generate_data_path = './test/'
test_data_path = './test/'
print("stage 0")
pp = PreProcessor(original_data_path+'train.csv', generate_data_path)
print("stage 1")
comments_id_content = pp.transform_data()
print("stage 2")
pp.generate_tokens(comments_id_content)
print("stage 3")

#gen = TfIdfGenerator(generate_data_path, generate_data_path + 'comments_id_tokens.csv')
#_ = gen.generate()
#gen = Word2Vector_Builder(generate_data_path, generate_data_path + 'comments_id_tokens.csv')
print("stage 4")
gen.bulid_word2vector()
print("stage 5")

c_v = comment_vector_builder("word2vec_wb.model", generate_data_path + 'comments_id_tokens.csv', 20)
c_v.generate(generate_data_path)
print('stage 6')
svm_m = Svm_Builder(generate_data_path + 'id_tokenv.csv', model_path)
svm_m.generate()
print('stage 7')
logistic_m = Logistic_Regression_Builder(generate_data_path + 'id_tokenv.csv', model_path)
logistic_m.generate()

pp_test = PreProcessor(test_data_path +'test.csv', test_generate_data_path)
comments_id_content = pp_test.transform_data(test = True)
print('stage 8')
pp_test.generate_tokens(comments_id_content, test = True)
print('stage 9')
c_v = comment_vector_builder("word2vec_wb.model", test_generate_data_path + 'comments_id_tokens.csv', 20)
c_v.generate(test_generate_data_path, test = True)
print('stage 10')
svm_m_t = Svm_Test_Builder(test_generate_data_path + "id_tokenv.csv", model_path + "svm.model", test_generate_data_path)
svm_m_t.generate()
logistic_t = Logistic_Test_Builder(test_generate_data_path + "id_token_score.csv", model_path + "logistic.model", test_generate_data_path)
logistic_t.generate()

