from utilities.DataUtils import DataUtils
import numpy as np
import pandas as pd
from learning.DataHolder import DataHolder
from learning.ID3 import IGClassifier
import datetime
from old_testing import get_words_matrix


def get_model():
    print("Started work on model: " + str(datetime.datetime.now()))
    X, y = DataUtils.preprocess_data('smallTest.txt')
    print("Finished pre-process: " + str(datetime.datetime.now()))
    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    print("Finished text strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    print("Finished vocabulary generation (" + str(data_holder.get_words_list_length()) + " words): " + str(datetime.datetime.now()))
    data_holder.sk_generate_reviews_words_matrix()
    print("Finished words matrix comprising: " + str(datetime.datetime.now()))
    Xy = np.append(data_holder.get_reviews_words_matrix(), data_holder.get_tagging_vector().reshape(-1, 1), axis=1)
    return Xy, data_holder


def main():

    # the new version
    #model = get_model()
    #words_matrix = model[0]


    # the old version:
    X, y = get_words_matrix()
    words_matrix = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)


    print('words matrix is')
    print(pd.DataFrame(words_matrix))
    ig_tree = IGClassifier(words_matrix)
    ig_tree.train()
    ig_tree.check_error()

if __name__ == "__main__":
    main()
