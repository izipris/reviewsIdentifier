from utilities.DataUtils import DataUtils
import numpy as np
from learning.DataHolder import DataHolder
from learning.RandomForest import RandomForest
import datetime


def get_model():
    print("Started work on model: " + str(datetime.datetime.now()))
    X, y = DataUtils.preprocess_data('COMMENTS_30K_REP.txt')
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


def run_random_forest_on_data(training_set, test_set):
    forest = RandomForest(850, training_set.astype(int), 7000, 3000)
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


def main():
    model = get_model()
    words_matrix = model[0]
    sets = DataUtils.split_to_sets(words_matrix, int(np.ma.size(words_matrix, axis=0) * 0.15))
    training_set = sets[0]
    test_set = sets[1]
    random_forest = run_random_forest_on_data(training_set, test_set)
    while True:
        s = input("Type a review about a cellphone: ")
        a = model[1].get_vectorizer().transform([s]).toarray()
        print("Prediction (1-Positive, 0-Negative): " + str(random_forest.predict(a[0])))


if __name__ == "__main__":
    main()
