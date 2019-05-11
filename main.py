# coding=utf-8

from preprocess import PreProcessor
from tf_idf_build import TfIdfGenerator
from word_vector_builder import Word2Vector_Builder
from getCommentVector import comment_vector_builder
from train_svm import Svm_Builder
import sys


original_data_path = './train/'
generate_data_path = './generate_datas/'
model_path = './models/'
test_generate_data_path = './test/'
test_data_path = './test/'
print("stage 0")
#pp = PreProcessor(original_data_path+'train.csv', generate_data_path)
print("stage 1")
#comments_id_content = pp.transform_data()
print("stage 2")
#pp.generate_tokens(comments_id_content)
print("stage 3")

#gen = TfIdfGenerator(generate_data_path, generate_data_path + 'comments_id_tokens.csv')
#_ = gen.generate()
#gen = Word2Vector_Builder(generate_data_path, generate_data_path + 'comments_id_tokens.csv')
print("stage 4")
#gen.bulid_word2vector()
print("stage 5")

#c_v = comment_vector_builder("word2vec_wb.model", generate_data_path + 'comments_id_tokens.csv', 20)
#c_v.generate(generate_data_path)
print('stage 6')
svm_m = Svm_Builder(generate_data_path + 'id_tokenv.csv', model_path)
svm_m.generate()
import sys
sys.exit(-1)
print('stage 7')
pp_test = PreProcessor(original_data_path+'test.csv', test_generate_data_path)
comments_id_content = pp_test.transform_data()
pp_test.generate_tokens(comments_id_content)





