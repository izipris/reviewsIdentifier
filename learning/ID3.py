# python file responsible for creating a classification tree using the ID3 algorithm

from learning.TreeNode import CustomTreeNode


def create_tree(X, y):
    root = CustomTreeNode(reviews=None)
