class Node:
    def __init__(self, feature_index, gini, left_child=None, right_child=None):
        self.__feature_index = feature_index
        self.__left_child = left_child
        self.__right_child = right_child
        self.__gini = gini

    def get_feature_index(self):
        return self.__feature_index

    def get_left_child(self):
        return self.__left_child

    def get_right_child(self):
        return self.__right_child

    def get_gini(self):
        return self.__gini

    def set_feature_index(self, feature_index):
        self.__feature_index = feature_index

    def set_left_child(self, node):
        self.__left_child = node

    def set_right_child(self, node):
        self.__right_child = node

    def set_gini(self, gini):
        self.__gini = gini