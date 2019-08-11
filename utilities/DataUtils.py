import csv
import random

from ftfy.badness import sequence_weirdness
import numpy as np


class DataUtils:
    """A library with general utilities for the project"""

    @staticmethod
    def is_string_encoding_corrupted(text):
        """Returns True iff the provided text contains corrupted characters"""
        return sequence_weirdness(text.encode('sloppy-windows-1252').decode('gb2312', 'replace')) > 0

    @staticmethod
    def tsv_file_to_np_array(data_file_path):
        """Gets a tsv file path and returns a data matrix"""
        X = []
        with open(data_file_path, 'r', encoding='sloppy-windows-1252') as f:
            reader = csv.reader(f, dialect='excel', delimiter='\t')
            for row in reader:
                X.append(row)
        return np.array(X)

    @staticmethod
    def remove_corrupted_data(X):
        """removes rows with corrupted data in the matrix"""
        delete_indices = []
        i = 0
        for row in X:
            if DataUtils.is_string_encoding_corrupted(row[0]): #or DataUtils.is_string_encoding_corrupted(row[1]):
                delete_indices.append(i)
            i += 1
        X = np.delete(X, delete_indices, axis=0)
        return X

    @staticmethod
    def split_data_tagging(data):
        """Returns the provided data matrix as (X,y) where X- data, y - tagging"""
        cols = np.ma.size(data, axis=1)
        return data[:, :cols - 1], data[:, cols - 1]

    @staticmethod
    def split_to_sets(data, test_set_size):
        data_size = np.ma.size(data, axis=0)
        test_indices = set(random.sample(range(data_size), test_set_size))
        train_indices = set(range(data_size)) - test_indices
        return np.take(data, list(train_indices), axis=0), np.take(data, list(test_indices), axis=0)



    @staticmethod
    def preprocess_data(data_file_path):
        """
        Pre-processing of the data. Expects for tab-delimited file with the following columns:
        1. Review title
        2. Review body
        3. Tag
        :param data_file_path: The path of the file with the data
        :return: A tuple (X,y) where X is data matrix and y is tagging vector
        """
        data = DataUtils.tsv_file_to_np_array(data_file_path)
        data = DataUtils.remove_corrupted_data(data)  # Remove corrupted rows
        data = np.delete(data, 0, axis=0)  # Remove headers
        return DataUtils.split_data_tagging(data)

