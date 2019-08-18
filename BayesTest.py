from utilities.DataUtils import DataUtils
import numpy as np
from learning.DataHolder import DataHolder
import datetime
from learning import adaboost
from learning import Bayes


def get_model():
    print("Started work on model: " + str(datetime.datetime.now()))
    X, y = DataUtils.preprocess_data('COMMENTS_30K_REP.txt')
    print("Finished pre-process: " + str(datetime.datetime.now()))
    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    print("Finished text strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    print("Finished vocabulary generation (" + str(
        data_holder.get_words_list_length()) + " words): " + str(
        datetime.datetime.now()))
    data_holder.sk_generate_reviews_words_matrix()
    print("Finished words matrix comprising: " + str(datetime.datetime.now()))
    Xy = np.append(data_holder.get_reviews_words_matrix(),
                   data_holder.get_tagging_vector().reshape(-1, 1), axis=1)
    return Xy, data_holder


update = False
if update:
    model = get_model()
    words_matrix = model[0]
    sets = DataUtils.split_to_sets(words_matrix, int(
        np.ma.size(words_matrix, axis=0) * 0.15))
    training_set = sets[0]
    test_set = sets[1]
    np.save('training', training_set)
    np.save('test', test_set)
else:
    training_set = np.load('training.npy')
    test_set = np.load('test.npy')

# ----big test bayes
bayes = Bayes.Classifier(training_set)
# test model
print("error: ", bayes.error(test_set[:, 0:-1], test_set[:, -1]))

# # ----small test bayes
# training = np.array([[1, 1, 0, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1],
#                      [0, 0, 1, 1, 0]])
# test = np.array([[1, 1, 1, 0, 1], [0, 0, 0, 1, 0]])
# single_test = np.array([1, 1, 1, 0, 1])
# bayes = Bayes.Classifier(training)
# # test model
# print("error: ", bayes.error(test[:, 0:-1], test[:, -1]))
# print('error for single: ', bayes.error(single_test[:-1], single_test[-1]))

##################

# adaboost
#
# for t in [1, 100, 400]:
#     print("---training ", t, "-boost ----")  #todo
#     boost = adaboost.AdaBoost(t)
#     print("created")  #todo
#     boost.train(training_set[:, 0:-1], training_set[:, -1])
#     print("error for ", t, ": ", boost.error(test_set[:, 0:-1], test_set[:,
#         -1]))


