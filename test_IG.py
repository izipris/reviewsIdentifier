from utilities.DataUtils import DataUtils
import numpy as np
import pandas as pd
from learning.DataHolder import DataHolder
from learning.ID3 import IGClassifier
import datetime
from old_testing import get_words_matrix
import matplotlib.pyplot as plt


def get_model(new, filename):
    print("Started work on model: " + str(datetime.datetime.now()))
    X, y = DataUtils.preprocess_data(filename)
    print("Finished pre-process: " + str(datetime.datetime.now()))
    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    print("Finished text strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    print("Finished vocabulary generation (" + str(data_holder.get_words_list_length()) + " words): " + str(datetime.datetime.now()))
    if new:
        data_holder.sk_generate_reviews_words_matrix()
    else:
        data_holder.generate_reviews_words_matrix()
        print("Finished words matrix comprising: " + str(datetime.datetime.now()))
        return (data_holder.get_reviews_words_matrix(), y), data_holder

    print("Finished words matrix comprising: " + str(datetime.datetime.now()))
    Xy = np.append(data_holder.get_reviews_words_matrix(), data_holder.get_tagging_vector().reshape(-1, 1), axis=1)
    return Xy, data_holder


def main():

    # the new version
    model = get_model(new=True, filename='COMMENTS_3.5K.txt')
    Xy = pd.DataFrame(model[0])


    # the old version - should be identical:
    #model = get_model(new=False, filename='COMMENTS_LESS.txt')
    #X = model[0][0]
    #y = model[0][1]
    #Xy = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)

    # the old version:
    #X, y = get_words_matrix('COMMENTS_LESS.txt')
    #Xy = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)


    train_Xy = Xy.sample(frac=0.5, random_state=151).reset_index(drop=True)  # random_state = 0
    test_Xy = Xy.drop(train_Xy.index).reset_index(drop=True)
    test_set = test_Xy.iloc[:, :-1]  # test set, no labels
    true_y = test_Xy.iloc[:, -1].tolist()  # true labels of test set

    print(test_Xy.loc[test_Xy[1752] == 1])
    x1= test_Xy.loc[test_Xy[1752] == 1]
    print('sum 1 is')
    print(x1.sum(axis = 0, skipna = True) )
    print('----')
    print(test_Xy.loc[test_Xy[1752] == 0])
    x2 = test_Xy.loc[test_Xy[1752] == 0]
    print('sum 0 is: ')
    print(x2.sum(axis=0, skipna=True))

    print('words matrix is')
    words_matrix = train_Xy
    print(words_matrix)
    exit(0)

    # plot train error as function of max depth todo- plot 3d function as function of traininfraction also!!
    # than plot as function of data set size
    n = 1
    errors = []
    for i in range(n):
        ig_tree = IGClassifier(words_matrix, training_fraction=0.5)
        print('start time: ', datetime.datetime.now())
        ig_tree.train()
        print('end time: ', datetime.datetime.now())

        label = ig_tree.predict(test_set)
        # calc error
        error = 0
        for i in range(len(label)):
            if int(label[i]) != int(true_y[i]):
                error += 1
        print('error before prune is: ', error / len(label))
        ig_tree.root.display()
        print('------now going to prune-------')

        ig_tree.prune()

        label = ig_tree.predict(test_set)
        # calc error
        error = 0
        for i in range(len(label)):
            if int(label[i]) != int(true_y[i]):
                error += 1
        print('error rate after prune is: ', error / len(label))
        errors.append(error/len(label))
        ig_tree.root.display()

    # importing the required module


    # x axis values
    x = list(range(1, len(errors)+1))
    # corresponding y axis values
    y = errors
    # plotting the points
    #plt.plot(x, y)
    #plt.xlabel('Max IG Tree Depth')
    #plt.ylabel('Error Rate')
    #plt.title('1300 sample data set, 2800 features \n Training fraction: 0.5')

    # function to show the plot
    #plt.show()


    # check on input:

    while True:
        s = input("Type a review about a cellphone: ")
        a = model[1].get_vectorizer().transform([s]).toarray()
        df = pd.DataFrame(a)
        a = a[0]
        print(a)
        for i in range(len(a)):
            if a[i] == 1:
                print(i, ' ', end='')
        print()


        print("Prediction (1-Positive, 0-Negative): " + str(ig_tree.predict(df)))

if __name__ == "__main__":
    main()
