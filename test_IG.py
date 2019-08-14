from utilities.DataUtils import DataUtils
import numpy as np
import pandas as pd
from learning.DataHolder import DataHolder
from learning.ID3 import IGClassifier
import datetime
from old_testing import get_words_matrix
import matplotlib.pyplot as plt


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


    # the old version to csv:
    X, y = get_words_matrix('smallTest.txt')
    words_matrix = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)

    #words_matrix.to_csv(path_or_buf='preprocess.csv')
    #from loaded preprocessed:
    #words_matrix = pd.read_csv('preprocess.csv')

    print('words matrix is')

    print(pd.DataFrame(words_matrix))

    # plot train error as function of max depth todo- plot 3d function as function of traininfraction also!!
    # than plot as function of data set size

    errors = []
    for i in range(20):
        ig_tree = IGClassifier(words_matrix, training_fraction=0.7, max_depth=i)
        ig_tree.train()
        err = ig_tree.check_error()
        errors.append(err)
        ig_tree.root.display()

    # importing the required module


    # x axis values
    x = list(range(20))
    # corresponding y axis values
    y = errors

    # plotting the points
    plt.plot(x, y)
    plt.xlabel('Max IG Tree Depth')
    plt.ylabel('Error Rate')
    plt.title('670 sample data set, 870 features \n Training fraction: 0.7')

    # function to show the plot
    plt.show()


    # check on input:

    #while True:
    #    s = input("Type a review about a cellphone: ")
    #    a = model[1].get_vectorizer().transform([s]).toarray()
    #    print(a)
    #    print(pd.DataFrame(a))
    #    print("Prediction (1-Positive, 0-Negative): " + str(ig_tree.predict(a)))

if __name__ == "__main__":
    main()
