from utilities.LanguageUtils import LanguageUtils
from utilities.DataUtils import DataUtils
import nltk
import numpy as np
from learning.DataHolder import DataHolder
from learning.Tree import Tree
import datetime
from sklearn import tree


def get_words_matrix():
    #print(datetime.datetime.now())
    X,y = DataUtils.preprocess_data('COMMENTS_LESS.txt')
    #X,y = DataUtils.preprocess_data('C:\\Users\\idzipris\\Downloads\\COMMENTS_10K.txt')
    #print("Finished pre-process: " + str(datetime.datetime.now()))

    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    #print("Finished strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    #print("Finished vocabulary (" + str(data_holder.get_words_list_length()) + "): " + str(datetime.datetime.now()))
    data_holder.generate_reviews_words_matrix()
    #print("Finished words matrix: " + str(datetime.datetime.now()))
    #print('Done: ' + str(datetime.datetime.now()))

    return data_holder.get_reviews_words_matrix(), y


X, y = get_words_matrix()
#print(X)
#print(y)

# training_set = Xy[0][0: 8000 ,  :], Xy[1][0: 8000]
# test_set = Xy[0][8000: 9500 ,  :], Xy[1][8000: 9500]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(training_set[0], training_set[1])
# for i in range(len(test_set[1])):
#     print(