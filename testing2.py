from utilities.LanguageUtils import LanguageUtils
from utilities.DataUtils import DataUtils
from learning import TreeNode
import nltk
import numpy as np
from learning.DataHolder import DataHolder
from learning.Tree import Tree
import datetime
from sklearn import tree

def get_words_matrix():
    print(datetime.datetime.now())
    #X,y = DataUtils.preprocess_data('C:\\Users\\Natan\\Desktop\\reviewsIdentifier\\testai.txt')
    X,y = DataUtils.preprocess_data('C:\\Users\\idzipris\\Downloads\\COMMENTS_10K.txt')
    print("Finished pre-process: " + str(datetime.datetime.now()))
    a = [1,1,1]
    b=[0,0,1,1]
    print(set(a) & set(b))

    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    print("Finished strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    print("Finished vocabulary (" + str(data_holder.get_words_list_length()) + "): " + str(datetime.datetime.now()))
    data_holder.generate_reviews_words_matrix()
    print("Finished words matrix: " + str(datetime.datetime.now()))
    print("Building Tree: " +  str(datetime.datetime.now()))
    tree1 = TreeNode.CustomTree(data_holder.get_tagging_vector(), data_holder.get_reviews_words_matrix())
    print("Finished Tree: " +  str(datetime.datetime.now()))
    ####
    # my test
    #####
    # matrix = np.matrix([[0,0,1],[1,0,0],[0,1,0],[1,0,0],[0,1,1],[1,1,1],[0,0,0]])
    # ratings = np.array([1,0,1,0,0,1,1])
    # tree1 = TreeNode.CustomTree(data_holder.get_tagging_vector(),data_holder.get_reviews_words_matrix())
    #####
    # end of my test
    #####
    print('Done: ' + str(datetime.datetime.now()))

    return 0




Xy = get_words_matrix()
# training_set = Xy[0][0: 8000 ,  :], Xy[1][0: 8000]
# test_set = Xy[0][8000: 9500 ,  :], Xy[1][8000: 9500]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(training_set[0], training_set[1])
# for i in range(len(test_set[1])):
#     print('Prediction: ' + str(clf.predict(test_set[0][i].reshape(1,-1))) + '\t Real: ' + str(test_set[1][i]))