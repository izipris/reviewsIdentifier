from utilities.DataUtils import DataUtils
import numpy as np
import pandas as pd
from learning.DataHolder import DataHolder
from learning.ID3 import IGClassifier
import datetime
import matplotlib.pyplot as plt


def save_attributes(outfile, attributes):
    with open(outfile, 'w') as f:
        for s in attributes:
            f.write(str(s) + '\n')


def load_attributes(infile):
    with open(infile, encoding="utf8", errors='ignore') as f:
        my_list = [line.rstrip('\n') for line in f]
    return my_list


def calc_error(pred_label, true_y):
    error = 0
    for i in range(len(pred_label)):
        if int(pred_label[i]) != int(true_y[i]):
            error += 1
    return error / len(pred_label)


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

    run_check = True  # check at end on our own input
    load_prev = True  # use saved preprocessed attribute list
    prune = False
    test_fraction = 0.5

    # the new version
    model = get_model(new=True, filename='COMMENTS_PARTIAL.txt')
    Xy = pd.DataFrame(model[0])
    Xy[Xy != 0] = 1  # make sure matrix is binary

    print('full matrix including test set is: ')
    print(Xy)

    # split to train and test data
    train_Xy = Xy.sample(frac=1-test_fraction, random_state=180).reset_index(drop=True)  # random_state = 0

    # select features with best information gain
    k = 1000  # num of attributes that are best
    if not load_prev:
        best_atts = IGClassifier.get_n_best_attributes_fast(k, Xy.iloc[:, :-1])  # select k best columns no lablel col
        print('best attributes are: ', best_atts)
        save_attributes('features.txt', best_atts)

    else:
        best_atts = list(map(lambda x: int(x), load_attributes('features.txt')))


    # plot train error as function num of selected features  out of k best
    # than plot as function of data set size
    errors = []
    pruned_errors = []
    for i in range(100, 400, 100):

        cur_best_atts = best_atts[:i]
        print('using ', len(cur_best_atts), 'attributes:' ,cur_best_atts)
        # use feature selection, select only best attibutes columns
        cols = cur_best_atts + [-1]
        words_matrix = train_Xy.iloc[:, cols]  # feature selection
        words_matrix.columns = range(words_matrix.shape[1])  # renames cols

        # strip irrelevent cols from test set
        test_Xy = Xy.drop(train_Xy.index).reset_index(drop=True).iloc[:, cols]
        test_Xy.columns = range(test_Xy.shape[1])

        #if i == 1:
        print('words matrix is')
        print(words_matrix)
        print('test set is:')
        print(test_Xy)

        # split test set for later with and without label
        test_set = test_Xy.iloc[:, :-1]  # test set, no labels
        true_y = test_Xy.iloc[:, -1].tolist()  # true labels of test set



        ig_tree = IGClassifier(words_matrix, training_fraction=1)  # no pruning this time
        print('start time: ', datetime.datetime.now())
        ig_tree.build()
        print('end time: ', datetime.datetime.now())

        label = ig_tree.predict(test_set)
        # calc error
        e = calc_error(label, true_y)
        errors.append(e)
        ig_tree.root.display()
        print('error is: ', e, 'i is: ', i)


        if prune:
            print('------now going to prune-------')
            ig_tree.prune()
            label = ig_tree.predict(test_set)
            e = calc_error(label, true_y)
            print('error rate after prune is: ', e)
            pruned_errors.append(e)
            ig_tree.root.display()


    # x axis values
    x = list(range(100, 400, 100))
    # corresponding y axis values
    y = errors
    y2 = pruned_errors
    # plotting the points
    plt.plot(x, y)

    plt.xlabel('Num of Selected Features for Tree Build')
    plt.ylabel('Test Error Rate')
    if prune:
        plt.plot(x, y2)
        plt.legend(['y = Pre Prune', 'y = Post Prune'], loc='upper left')
    plt.title('10,000 total sample data set, 6500 features Generated \n Training Set fraction: 0.75 \n no Pruning')

    # function to show the plot
    plt.show()

    # run input check at end
    if run_check:
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


    # old version
    #model = get_model(new=False, filename='COMMENTS_PARTIAL.txt')
    #X = model[0][0]
    #y = model[0][1]
    #Xy = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)