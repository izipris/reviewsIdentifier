import math
import sys
import datetime
import numpy as np
import pandas as pd
import nltk

# pre-installation
try:
    nltk.data.find('stopwords')
    nltk.data.find('punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
from learning.DataHolder import DataHolder
from learning.RandomForest import RandomForest
from utilities.DataUtils import DataUtils
from learning.ID3 import IGClassifier
from learning import Bayes

# Constants
ERROR_PROVIDE_SAMPLE = "Please provide a path for a data .tsv file in the 1st argument!"
INPUT_MSG_ALGO = "Please select an algorithm ('F'-Random Forest, 'I' - Information Gain, 'N' - Naive Bayes: "
VALID_ALGOS = ['F', 'I', 'N']
TEST_SET_PCTG = 0.15


def get_data_holder(data_file_path):
    """
    Generates a DataHolder object for the data in the provided path
    :param data_file_path: *.tsv data file, where first n-1 columns are features and n'th column is tagging
    :return: (Xy matrix, DataHolder object)
    """
    print("Started work on model: " + str(datetime.datetime.now()))
    X, y = DataUtils.preprocess_data(data_file_path)
    print("Finished pre-process: " + str(datetime.datetime.now()))
    data_holder = DataHolder(X, y)
    data_holder.reviews_strip()
    print("Finished text strip: " + str(datetime.datetime.now()))
    data_holder.extract_vocabulary()
    print("Finished vocabulary generation (" + str(data_holder.get_words_list_length()) + " words): " + str(
        datetime.datetime.now()))
    data_holder.sk_generate_reviews_words_matrix()
    print("Finished words matrix comprising: " + str(datetime.datetime.now()))
    return data_holder


def run_random_forest_on_data(data_holder_words_matrix, num_of_trees, num_of_features, num_of_samples):
    """Gets a words matrix of DataHolder object, builds a random forest on a training set,
    calculates accuracy for a test set, and returns the forest object."""
    training_set, test_set = DataUtils.split_to_sets(
        words_matrix, int(np.ma.size(data_holder_words_matrix, axis=0) * TEST_SET_PCTG))
    forest = RandomForest(num_of_trees, training_set.astype(int), num_of_samples, num_of_features)
    print("Started to build the Random Forest: " + str(datetime.datetime.now()))
    forest.build()
    print("Finished to build the Random Forest: " + str(datetime.datetime.now()))
    print("Calculating accuracy of the Random Forest...")
    good_prediction_counter = 0
    n = 0
    for sample in list(test_set):
        prediction = forest.predict(sample[:len(sample) - 1])
        if prediction == int(sample[-1]):
            good_prediction_counter += 1
        n += 1
    print("Random Forest Accuracy: " + str(good_prediction_counter / np.ma.size(test_set, axis=0)))
    return forest


def run_information_gain_on_data(data_holder_words_matrix, k_best_attrubutes, prune_fraction, prune, max_depth,
                                 offline=False):
    """Gets a words matrix of DataHolder object, builds a tree with IG on a training set,
    calculates accuracy for a test set, and returns the tree object."""
    TEST_FRACTION = 0.15

    Xy = pd.DataFrame(data_holder_words_matrix)
    Xy[Xy != 0] = 1  # make sure matrix is binary
    train_Xy = Xy.sample(frac=1 - TEST_FRACTION).reset_index(drop=True)
    best_atts = IGClassifier.get_n_best_attributes_fast(k_best_attrubutes,
                                                        Xy.iloc[:, :-1])  # select k best columns no lablel col

    cols = best_atts + [-1]
    words_matrix = train_Xy.iloc[:, cols]  # feature selection
    test_Xy = Xy.drop(train_Xy.index).reset_index(drop=True).iloc[:, cols]
    test_set = test_Xy.iloc[:, :-1]  # test set, no labels
    true_y = test_Xy.iloc[:, -1].tolist()  # true labels of test set

    if max_depth < 0:
        max_depth = math.inf
    ig_tree = IGClassifier(words_matrix, max_depth=max_depth, training_fraction=1 - prune_fraction)
    print("Started to build the IG: " + str(datetime.datetime.now()))
    ig_tree.build()
    print("Finished to build the IG: " + str(datetime.datetime.now()))
    print("Calculating accuracy of the IG...")
    label = ig_tree.predict(test_set)
    # calc error
    e = IGClassifier.calc_error(label, true_y)

    print("IG Accuracy: " + str((1 - float(e))))
    if prune:
        print("Now pruning...")
        ig_tree.prune()
        label = ig_tree.predict(test_set)
        e = IGClassifier.calc_error(label, true_y)

        print("IG w/ Pruning Accuracy: " + str((1 - float(e))))
    return ig_tree


def run_bayes_on_data(data_holder_words_matrix):
    """Gets a words matrix of DataHolder object, builds Naive Bayes Classifier on a training set,
    calculates accuracy for a test set, and returns the Classifier object."""
    training_set, test_set = DataUtils.split_to_sets(
        words_matrix, int(np.ma.size(data_holder_words_matrix, axis=0) * TEST_SET_PCTG))
    training_set = training_set.astype(np.float)
    test_set = test_set.astype(np.float)
    print("Started to build the Naive Bayes Classifier: " + str(datetime.datetime.now()))
    bayesB = Bayes.Classifier(training_set, prune=True, short_prune=True)
    print("Finished to build the Random Forest: " + str(datetime.datetime.now()))
    print("Calculating accuracy of the Naive Bayes Classifier...")
    e = bayesB.error(test_set[:, 0:-1], test_set[:, -1])
    print("Naive Bayes Classifier Accuracy: " + str((1 - float(e))))
    return bayesB


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(ERROR_PROVIDE_SAMPLE)
        exit(-1)
    # Organize training set & test set
    test_data_file = sys.argv[1]
    data_holder = get_data_holder(test_data_file)
    words_matrix = data_holder.get_Xy_matrix()
    # Get user's decision
    input_algorithm = input(INPUT_MSG_ALGO)
    while input_algorithm not in VALID_ALGOS:
        input_algorithm = input(INPUT_MSG_ALGO)
    # Fit the chosen algorithm
    model = None
    if input_algorithm == 'F':  # Random Forest
        num_of_trees = input("Number of trees in the forest: ")
        num_of_features = input("Max number of features in a tree: ")
        num_of_samples = input("Number of samples in a tree: ")
        model = run_random_forest_on_data(words_matrix, int(num_of_trees), int(num_of_features), int(num_of_samples))
    elif input_algorithm == 'I':  # IG
        print("total num of features: ", len(words_matrix[0]))
        num_of_features = int(
            input("choose how many features you want to train on (must be less than total num of features) "))
        prune = int(input("would you like to prune at the end? (1 - True, 0 - False) "))
        prune_fraction = 0
        if prune:
            prune_fraction = float(input("choose fraction of hold out data for prune (number: 0 < input < 1) "))

        max_depth = int(input("limit tree to max depth? (-1 not to, any integer to pick max depth) "))
        model = run_information_gain_on_data(words_matrix, num_of_features, prune_fraction, prune, max_depth)

    elif input_algorithm == 'N':  # Naive Bayes
        model = run_bayes_on_data(words_matrix)
    # Let user play
    while True:
        review = input("Type a review about a cellphone (or 'q' to quit): ")
        if review == 'q':
            exit(0)
        model_prediction = data_holder.get_vectorizer().transform([review]).toarray()
        if input_algorithm == 'I':
            print("Prediction (1-Positive, 0-Negative): " + str(model.predict(model_prediction)[0]))
        else:
            print("Prediction (1-Positive, 0-Negative): " + str(model.predict(model_prediction[0])))
