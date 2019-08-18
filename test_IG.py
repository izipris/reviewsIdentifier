import pickle

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

    # the new version
    model = get_model(new=True, filename='COMMENTS_PARTIAL.txt')
    Xy = pd.DataFrame(model[0])
    Xy[Xy != 0] = 1  #   -------is thus the fix??

    print('full matrix including test set is: ')
    print(Xy)

    train_Xy = Xy.sample(frac=0.75, random_state=100).reset_index(drop=True)  # random_state = 0  # take half for train

    k = 1000  # num of attributes that are best
    best_atts = IGClassifier.get_n_best_attributes_fast(k, Xy.iloc[:, :-1])  # select k best columns no lablel col
    print('best attributes are: ', best_atts)
    save_attributes('features.txt', best_atts)

    #best_atts = list(map(lambda x: int(x), load_attributes('features.txt')))
    #print(best_atts)
    exit(0)

    cols = best_atts + [-1]
    words_matrix = train_Xy.iloc[:, cols]  # feature selection
    words_matrix.columns = range(words_matrix.shape[1])  # renames cols

    test_Xy = Xy.drop(train_Xy.index).reset_index(drop=True).iloc[:, cols]
    test_Xy.columns = range(test_Xy.shape[1])

    print('words matrix is')
    print(words_matrix)
    print('test set is:')
    print(test_Xy)

    #create test set for later
    test_set = test_Xy.iloc[:, :-1]  # test set, no labels
    true_y = test_Xy.iloc[:, -1].tolist()  # true labels of test set

    # plot train error as function of max depth todo- plot 3d function as function of traininfraction also!!
    # than plot as function of data set size
    n = 50
    errors = []
    pruned_errors = []
    for i in range(1, n, 4):
        ig_tree = IGClassifier(words_matrix, max_depth=i, training_fraction=0.25)
        print('start time: ', datetime.datetime.now())
        ig_tree.train()
        print('end time: ', datetime.datetime.now())

        label = ig_tree.predict(test_set)
        # calc error
        e = calc_error(label, true_y)
        errors.append(e)
        print('error before prune is: ', e)
        ig_tree.root.display()
        print('------now going to prune-------')

        ig_tree.prune()

        label = ig_tree.predict(test_set)
        e = calc_error(label, true_y)
        print('error rate after prune is: ', e)
        pruned_errors.append(e)
        ig_tree.root.display()
        print(IGClassifier.get_attributes_from_tree(ig_tree.root))

    # importing the required module


    # x axis values
    x = list(range(1, n, 4))
    # corresponding y axis values
    y = errors
    y2 = pruned_errors
    # plotting the points
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.xlabel('max depth')
    plt.ylabel('Test Error Rate')
    plt.legend(['y = Pre Prune', 'y = Post Prune'], loc='upper left')
    plt.title('10,000 total sample data set, 1000 features selected \n Training Set fraction: 0.75 \n 75% of that used for holdout data for pruning')

    # function to show the plot
    plt.show()


    # run input check at end
    run_check = True
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




    ##### 'phone' has like 360 bad apperances as 120 good appearnces in comments 3.5k ###### todo- for if i want to test with ratio



    # old version
    #model = get_model(new=False, filename='COMMENTS_PARTIAL.txt')
    #X = model[0][0]
    #y = model[0][1]
    #Xy = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)