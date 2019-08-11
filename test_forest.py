from utilities.DataUtils import DataUtils
import numpy as np
from learning.DataHolder import DataHolder
from learning.RandomForest import RandomForest
import datetime


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
    print('Done: ' + str(datetime.datetime.now()))
    return Xy


Xy = get_words_matrix()
sets = DataUtils.split_to_sets(Xy, int(np.ma.size(Xy, axis=0) * 0.15))
training_set = sets[0]
test_set = sets[1]
forest = RandomForest(50, training_set.astype(int))
forest.build()
print("Finished Forest: " + str(datetime.datetime.now()))
counter = 0
n = 0
for sample in list(test_set):
    prediction = forest.predict(sample[:len(sample) - 1])
    if prediction == int(sample[-1]):
        counter += 1
    print("predicted for sample " + str(n + 1) + ": " + str(prediction) + ", and it's really: " + str(int(sample[-1])))
    n += 1
print("Accuracy: " + str(counter / np.ma.size(test_set, axis=0)))