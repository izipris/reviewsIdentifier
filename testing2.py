from utilities.LanguageUtils import LanguageUtils
from utilities.DataUtils import DataUtils
import nltk
import numpy as np
from learning.DataHolder import DataHolder
from learning.Tree import Tree
import datetime
from sklearn import tree

def get_words_matrix():
    print(datetime.datetime.now())
    #X,y = DataUtils.preprocess_data('C:\\Users\\idzipris\\Downloads\\testai.txt')
    X,y = DataUtils.preprocess_data('C:\\Users\\idzipris\\Downloads\\COMMENTS_20K_REP.txt')
    print("Finished pre-process: " + str(datetime.datetime.now()))
    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    print("Finished strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    print("Finished vocabulary (" + str(data_holder.get_words_list_length()) + "): " + str(datetime.datetime.now()))
    data_holder.sk_generate_reviews_words_matrix()
    print("Finished words matrix: " + str(datetime.datetime.now()))
    Xy = np.append(data_holder.get_reviews_words_matrix(), data_holder.get_tagging_vector().reshape(-1, 1), axis=1)

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

    return Xy



#
# matrix = np.array([[0,0,1],[1,0,0],[0,1,0],[1,0,0],[0,1,1],[1,1,1],[0,0,0]])
# t = Tree(matrix, 2)
# t.build()

Xy = get_words_matrix()
sets = DataUtils.split_to_sets(Xy, int(np.ma.size(Xy, axis=0) * 0.15))
training_set = sets[0]
test_set = sets[1]
trees = []
while True:
    #num = int(input("number of trees: "))
    #print("Building Forest: " + str(datetime.datetime.now()))
    for i in range(int(50)): # try on 300
        t = Tree(training_set.astype(int))
        t.build()
        trees.append(t)
        print("Finished tree of " + str(t.get_number_of_features()) + " features: " + str(i))
    print("Finished Forest: " + str(datetime.datetime.now()))
    counter = 0
    n=0
    for sample in list(test_set):
        t_yes = 0
        t_no = 0
        for t in trees:
            if t.predict(sample[:len(sample) - 1]) == 1:
                t_yes += 1
            else:
                t_no += 1
        answer = 1 if t_yes >= t_no else 0
        if answer == int(sample[-1]):
            counter += 1
        print("predicted for sample " + str(n + 1) + ": " + str(answer) + ", and it's really: " + str(int(sample[-1])) + ". #yes: " + str(t_yes) + ", #no: " + str(t_no))
        n+=1
    print("Accuracy: " + str(counter / np.ma.size(test_set, axis=0)))
# while True:
#     num = int(input("number: "))
#     print("Building Tree: " +  str(datetime.datetime.now()))
#     tree1 = Tree(Xy.astype(int),num)
#     tree1.build()
#     print("Finished Tree: " +  str(datetime.datetime.now()))


# training_set = Xy[0][0: 8000 ,  :], Xy[1][0: 8000]
# test_set = Xy[0][8000: 9500 ,  :], Xy[1][8000: 9500]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(training_set[0], training_set[1])
# for i in range(len(test_set[1])):
#     print('Prediction: ' + str(clf.predict(test_set[0][i].reshape(1,-1))) + '\t Real: ' + str(test_set[1][i]))