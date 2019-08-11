import random
import numpy as np
from learning.Node import Node


class Tree:
    def __init__(self, Xy, max_num_of_samples=None, max_num_of_features=None):
        """
        Tree constructor.
        Methodology: https://www.youtube.com/watch?v=7VeUPuFGJHk
        :param Xy: A data matrix where (m-1) first columns are features, and m column is tagging
        """
        self.__Xy = Xy
        # Select randomly samples
        if max_num_of_samples is None:
            max_num_of_samples = random.randint(3, int(np.ma.size(Xy, axis=0)))
        self.__random_pick_of_data = Xy[np.random.choice(Xy.shape[0], max_num_of_samples, replace=False), :]
        # Select randomly features
        if max_num_of_features is None:
            max_num_of_features = int(np.ma.size(self.__random_pick_of_data, axis=1) / 2)
        self.__features = random.sample(population=range(0, np.ma.size(self.__random_pick_of_data, axis=1) - 1),
                                        k=random.randint(3, max_num_of_features))
        self.__root = None

    def build(self):
        self.__root = self.generate_node(self.__features, self.__random_pick_of_data, None)

    def predict(self, sample):
        node = self.__root
        feature = node.get_feature_index()
        while not isinstance(feature, bool):
            if sample[feature] == 1:
                node = node.get_left_child()
            else:
                node = node.get_right_child()
            feature = node.get_feature_index()
        if feature:
            return 1
        return 0

    def generate_node(self, features_indices, data, parent):
        if np.ma.size(data, axis=0) == 0:
            return None
        if len(features_indices) < 1:  # No more features to separate
            if np.ma.size(data[data[:, -1] == 1], axis=0) >= np.ma.size(data[data[:, -1] == 0], axis=0):
                return Node(True, -1)
            return Node(False, -1)
        gini_of_features = self.calculate_gini_of_features(data)
        feature_with_min_gini = features_indices[np.argmin(np.take(gini_of_features, features_indices))]
        if parent is not None:
            if np.min(np.take(gini_of_features, features_indices)) >= parent.get_gini():  # Parent has lower gini
                return None
        node = Node(feature_with_min_gini, np.min(np.take(gini_of_features, features_indices)))
        next_feature_indices = [x for x in features_indices if x != feature_with_min_gini]
        data_feature_yes = data[data[:, feature_with_min_gini] == 1]
        data_feature_no = data[data[:, feature_with_min_gini] == 0]
        node.set_left_child(self.generate_node(next_feature_indices, data_feature_yes, node))
        node.set_right_child(self.generate_node(next_feature_indices, data_feature_no, node))
        if node.get_left_child() is None:
            if np.ma.size(data_feature_yes[data_feature_yes[:, -1] == 1], axis=0) >= np.ma.size(data_feature_yes[data_feature_yes[:, -1] == 0], axis=0):
                node.set_left_child(Node(True, -1))
            else:
                node.set_left_child(Node(False, -1))
        if node.get_right_child() is None:
            if np.ma.size(data_feature_no[data_feature_no[:, -1] == 1], axis=0) >= np.ma.size(data_feature_no[data_feature_no[:, -1] == 0], axis=0):
                node.set_right_child(Node(True, -1))
            else:
                node.set_right_child(Node(False, -1))
        return node

    def calculate_gini_of_features(self, Xy):
        tagged_true = Xy[Xy[:, -1] == 1]
        tagged_false = Xy[Xy[:, -1] == 0]
        feature_1_yes = np.sum(tagged_true, axis=0)
        feature_0_yes = np.ma.size(tagged_true, axis=0) - feature_1_yes
        feature_1_no = np.sum(tagged_false, axis=0)
        feature_0_no = np.ma.size(tagged_false, axis=0) - feature_1_no
        feature_1_gini = 1 - np.power(feature_1_yes / (feature_1_yes + feature_1_no), 2) - np.power(feature_1_no / (feature_1_yes + feature_1_no), 2)
        feature_0_gini = 1 - np.power(feature_0_yes / (feature_0_yes + feature_0_no), 2) - np.power(feature_0_no / (feature_0_yes + feature_0_no), 2)
        feature_final_gini = (np.where(np.isnan(feature_1_gini), 1, feature_1_gini) * ((feature_1_yes + feature_1_no) / (feature_1_yes + feature_1_no + feature_0_yes + feature_0_no))) + \
                             (np.where(np.isnan(feature_0_gini), 1, feature_0_gini) * ((feature_0_yes + feature_0_no) / (feature_1_yes + feature_1_no + feature_0_yes + feature_0_no)))
        return feature_final_gini

    def get_number_of_features(self):
        return len(self.__features)

