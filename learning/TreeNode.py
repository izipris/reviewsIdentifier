"""
This class makes an efficient binary decision tree.

It was checked for the matrix
the matrix    the rating
0 | 0 | 1       1
1 | 0 | 0       0
0 | 1 | 0       1
1 | 0 | 0       0
0 | 1 | 1       0
1 | 1 | 1       1
0 | 0 | 0       1

I did not implement the predict method.
The names I chose are not great and you are welcome to change them accordingly.
"""

import random
import math
import numpy as np


class CustomTreeNode:
    """
    this node has to keep track of the path and the keywords in the track
    because each node is constructed based on the node above it.
    """

    def __init__(self, matrix, data=None, pos=None, neg=None):
        self.data = data
        self.positive_answer = pos
        self.negative_answer = neg
        self.reviews = matrix


class CustomTree:
    def __init__(self, rf_matrix):
        self.tagging = None
        self.features = None
        self.data = None
        self.__root = self.build_tree(rf_matrix)


    def rate(self, review):
        """
        This gets a list of keywords and identifies which rating, the review
        should receive using the tree.
        :param review: the features that the review has and does not have
        :type review: list
        :return: the rating that is diagnosed
        """
        node = self.__root

        while node.positive_answer is not None:  # going through the tree until
            # getting to a leaf that has no children.

            if review[node.dat] == 1:
                node = node.positive_answer

            else:
                node = node.negative_answer

        return node.data


    def build_tree(self, rf_matrix):
        """
        Builds a tree
        :param rf_matrix: a matrix with words and reviews
        :return: the root of the tree built
        """
        # zeroArray = np.zeros(len(tag_list), dtype=int)
        self.select_randomly(rf_matrix)
        positive_reviews, negative_reviews = calculate_positive_and_negative(self.tagging)
        node = self.add_custom_node(positive_reviews, negative_reviews, self.features,
                                    self.data)

        return node


    def select_randomly(self, rf_matrix):
        """
                Tree constructor.
                Methodology: https://www.youtube.com/watch?v=7VeUPuFGJHk
                :param Xy: A data matrix where (m-1) first columns are features, and m column is tagging
                """
        num_of_samples = random.randint(3, int(np.ma.size(rf_matrix, axis=0) / 2))
        # Select randomly samples
        random_pick_of_data = rf_matrix[np.random.choice(rf_matrix.shape[0], num_of_samples, replace=False), :]
        data_num_of_columns = np.ma.size(random_pick_of_data, axis=1)
        self.data = random_pick_of_data[:, :data_num_of_columns - 1]
        self.tagging = random_pick_of_data[:, data_num_of_columns - 1]
        # Select randomly features
        self.features = random.sample(population=range(0, np.ma.size(self.data, axis=1)),
                                        k=random.randint(1, int(np.ma.size(self.data, axis=1) / 2)))


    def add_custom_node(self, good_tags, bad_tags, features, rf_matrix, pure=False):
        """
        makes a node for the tree
        :param good_tags: a list of good tag indices
        :param bad_tags:  a list of bad tag indices
        :param reviews:  a list of indices that are to be calculated in the tree
        :param features: a list of indices representing the features that are in the tree
        :param rf_matrix: a review feature matrix
        :param pure: an indicator if this is definitely a leaf node
        :return: the node to the root of the tree
        """
        node = CustomTreeNode(rf_matrix)

        ######################
        # this is the random forest ree that he had done in the video of random forest
        #######
        # sample two features
        # if len(features) >= 2:
        #     # randomFeatures = random.sample(features, 2)
        #     feature_used, positive_reviews, negative_reviews, feature_impurity = calculate_efficiency(
        #         randomFeatures, reviews, good_tags, bad_tags, rf_matrix)
        #     # features.remove(randomFeatures)
        #     features = [ feat for feat in features if feat not in randomFeatures]

        ######################

        # In order to continue building the tree there has to be a decrease in impurity
        if not pure and len(features) > 0:
            feature_used, positive_reviews, negative_reviews, feature_impurity = \
                self.calculate_efficiency(
                features, good_tags, bad_tags, rf_matrix)

            positive_reviews = list(positive_reviews)
            negative_reviews = list(negative_reviews)

            if feature_impurity < calc_node_gini(
                 len(good_tags), len(bad_tags), len(positive_reviews) + len(negative_reviews)):
                pure = False
                if feature_impurity == 0:
                    pure = True

                node.data = feature_used

                # adding a positive subtree
                node.positive_answer = self.add_custom_node(set(good_tags) & set(positive_reviews),
                                                            set(bad_tags) & set(positive_reviews),
                                                       [item for item in features if item !=
                                                        feature_used],
                                                            rf_matrix[positive_reviews, :], pure)

                # adding a negative subtree
                node.negative_answer = self.add_custom_node(set(good_tags) & set(negative_reviews),
                                                            set(bad_tags) & set(negative_reviews),
                                                       [item for item in features if item !=
                                                        feature_used], rf_matrix[
                                                                negative_reviews, :], pure)

                return node

        # adding leaves to the tree
        good_reviews = len(good_tags)
        bad_reviews = len(bad_tags)
        if good_reviews > bad_reviews:
            node.data = 1
        else:
            node.data = 0

        return node


    def calculate_efficiency(self, features, good_tags, bad_tags, rf_matrix):
        """
        This calculates which feature is the most efficient feature to use
        :param features: a list of all indices of optional features
        :param reviews: a list of all review indices
        :param good_tags:  a list of good tag indices
        :param bad_tags:  a list of bad tag indices
        :param rf_matrix: a review feature matrix
        :return: 4 things - the most efficient feature, the nodes that have that feature, the nodes
        that do not,
        the efficiency rating
        """
        feature_impurity = math.inf
        feature = -1
        positive_reviews = set()
        negative_reviews = set()

        for i in features:
            column = rf_matrix[:, i]
            positive_matches, negative_matches = calculate_positive_and_negative(column)

            # the indices of tags that are positive features
            # positive_feature = set(positive_matches) & set(reviews)
            positive_feature_positive_review = set(positive_matches) & set(good_tags)
            positive_feature_negative_review = set(positive_matches) & set(bad_tags)

            # the indices of negative feature
            # negative_feature = set(negative_matches) & set(reviews)
            negative_feature_positive_review = set(negative_matches) & set(good_tags)
            negative_feature_negative_review = set(negative_matches) & set(bad_tags)

            impurity = calc_impurity(positive_feature_positive_review, positive_feature_negative_review,
                                     negative_feature_positive_review, negative_feature_negative_review,
                                     len(positive_matches) + len(negative_matches))
            if impurity < feature_impurity:
                positive_reviews = positive_matches
                negative_reviews = negative_matches
                feature_impurity = impurity
                feature = i

        return feature, positive_reviews, negative_reviews, feature_impurity


def calc_impurity(pfpr, pfnr, nfpr, nfnr, total_amount):
    """
    Calculates the inputiry of a single node
    :param pfpr: the positive feature positive review list
    :param pfnr: the positive feature negative review list
    :param nfpr: the negative feature positive review list
    :param nfnr: the negative feature negative review list
    :param total_amount: the total amount of reviews in the node
    :return: the impurity rating
    """
    number_of_positives = len(pfnr) + len(pfpr)
    number_of_negatives = len(nfnr) + len(nfpr)
    positive_gini = calc_node_gini(len(pfpr), len(pfnr), number_of_positives)
    negative_gini = calc_node_gini(len(nfpr), len(nfnr), number_of_negatives)
    return (number_of_positives * positive_gini + number_of_negatives * negative_gini)/total_amount


def calc_node_gini(positive_amount, negative_amount, total_amount):
    """
    A helper function to calculate the gini for a node
    :param positive_amount: the amount of positive reviews
    :param negative_amount: the amount of negativre reviews
    :param total_amount: the total amount
    :return: the gini for the node
    """
    if total_amount == 0:
        return 0
    return 1 - math.pow(positive_amount/total_amount, 2) - math.pow(negative_amount/total_amount, 2)


def calculate_positive_and_negative(array):
    """
    Calculates the positive indices that have 1 and those that have a 0
    :param array: the array tha is to be seen
    :return: the positive indices, and the negative indices
    """
    new_array = np.ravel(array)
    x = new_array.astype(np.int)
    positive = np.flatnonzero(x)
    negative = set(np.arange(0, len(array))) - set(positive)
    return positive, negative