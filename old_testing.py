from utilities.LanguageUtils import LanguageUtils
from utilities.DataUtils import DataUtils
import nltk
import numpy as np
from learning.DataHolder import DataHolder
from learning.Tree import Tree
import datetime
from sklearn import tree
# todo- cross validation to reduce overfitting
# pruning - once tree is ready take away leafs by going over each leaf removing it and checking results
# model selection? feature selection -- try limiting depth


def get_words_matrix(filename):
    print(datetime.datetime.now())
    X,y = DataUtils.preprocess_data(filename)
    #X,y = DataUtils.preprocess_data('C:\\Users\\idzipris\\Downloads\\COMMENTS_10K.txt')
    print("Finished pre-process: " + str(datetime.datetime.now()))

    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    print("Finished strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    print("Finished vocabulary (" + str(data_holder.get_words_list_length()) + "): " + str(datetime.datetime.now()))
    data_holder.generate_reviews_words_matrix()
    print("Finished words matrix: " + str(datetime.datetime.now()))
    print('Done: ' + str(datetime.datetime.now()))

    return data_holder.get_reviews_words_matrix(), y


with open('COMMENTS_30K_REP.txt', encoding="utf8", errors='ignore') as f:
    lines = f.readlines()
desired_lines = lines[1::3]

with open('COMMENTS_PARTIAL.txt', 'w') as f:
    for l in desired_lines:
        f.write("%s" % l)

# the old version - should be identical:
#model = get_model(new=False, filename='COMMENTS_LESS.txt')
#X = model[0][0]
#y = model[0][1]
#Xy = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)

# the old version:
#X, y = get_words_matrix('COMMENTS_LESS.txt')
#Xy = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)