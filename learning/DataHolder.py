from utilities.LanguageUtils import LanguageUtils
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class DataHolder:
    def __init__(self, X, y):
        self.__reviews_vector = X
        self.__tagging_vector = y.astype(int)
        self.__words_list = []
        self.__reviews_words_matrix = None
        self.__vocab_dict = {}
        self.__sk_reviews_vector = []
        self.__vectorizer = None

    def reviews_strip(self):
        """For each review - move to lower case, remove: numbers, punctuation, stop words, contractions"""
        for row in self.__reviews_vector:
            row[0] = LanguageUtils.lemmatize_sentence(LanguageUtils.to_lower(LanguageUtils.remove_punctuation(LanguageUtils.remove_numbers(
                LanguageUtils.remove_stop_words(LanguageUtils.remove_contractions(row[0]))))))

    def extract_vocabulary(self):
        """Extract all the words of the reviews and comprise words vector"""
        vocabulary = set()
        for x in self.__reviews_vector:
            tokenized = nltk.word_tokenize(x[0])
            for w in tokenized:
                if w in self.__vocab_dict:
                    self.__vocab_dict[w] += 1
                else:
                    self.__vocab_dict[w] = 1
        for k in self.__vocab_dict:
            if self.__vocab_dict[k] > 3 and 2 < len(k) < 12:
                vocabulary.add(k)
        self.__words_list = sorted(list(vocabulary))
        # Optional for sk_generate_reviews_words_matrix:
        for x in self.__reviews_vector:
            tokenized = nltk.word_tokenize(x[0])
            for w in tokenized:
                if w not in self.__words_list:
                    tokenized.remove(w)
            self.__sk_reviews_vector.append(' '.join(tokenized))

    def generate_reviews_words_matrix(self):
        """Generate a matrix where each row is a vector which represents a review by the words vector"""
        final = np.array([[]]*len(self.__reviews_vector))
        count = 0
        for word in self.__words_list:
            tmp = np.char.find(self.__reviews_vector, word).reshape(-1, 1)
            final = np.append(final, tmp.reshape(-1, 1), axis=1)
            #if count % 50 == 0:
            #    print(str(count))
            count += 1
        final[final >= 0] = 1
        final[final < 0] = 0
        self.__reviews_words_matrix = final

    def sk_generate_reviews_words_matrix(self):
        self.__vectorizer = CountVectorizer()
        X = self.__vectorizer.fit_transform(self.__sk_reviews_vector)
        self.__reviews_words_matrix = np.array(X.toarray())


    def get_reviews_words_matrix(self):
        """Returns the matrix of the reviews represented by the words vector"""
        return self.__reviews_words_matrix

    def get_words_list_length(self):
        """Returns the length of the words vector"""
        return len(self.__words_list)

    def get_tagging_vector(self):
        return self.__tagging_vector

    def get_vectorizer(self):
        return self.__vectorizer
