# python file responsible for creating a classification tree using the ID3 algorithm
import pandas as pd
import numpy as np
from learning.TreeNode import CustomTreeNode
from testing import get_words_matrix
import math


class IGNode:

    def __init__(self, attribute=None, label=None, one=None, zero=None):
        self.attribute = attribute  # the attribute this node splits by
        self.label = label  # in case this node is a leaf it has no attribute but only a label
        self.one = one  # subtree for value is one
        self.zero = zero  # subtree for value is zero

    def __str__(self):
        h = self.get_height()

        if h == 0:
            # its a leaf
            return 'leaf label: ' + str(self.label)

        assert self.attribute is not None
        return str(self.attribute) + '--->' + str(self.one) + '|||' +  str(self.zero)
        #return ' ' * 2 * h + str(self.attribute) + '\n' + ' ' * 2 * (h - 1) + '/' +


    def get_height(self):
        h = 0
        if self.one is not None:
            h = self.one.get_height() + 1

        if self.zero is not None:
            h_temp = self.zero.get_height() + 1
            if h_temp > h:
                h = h_temp

        return h





def tree_builder(df, node):
    """recursive function"""
    status = df_leaf_status(df)

    if type(status) == int:
        assert status is not None
        node.label = status
        return
    elif status == 0 or status == 1:  # todo -- get rid of this
        print('bug!!!')
        exit(1)

    a = get_max_IG_attr(df)
    assert a is not None
    # now for each value, 0 or 1 that was in
    node.attribute = a
    df_one = df.loc[df[a] == 1]
    df_zero = df.loc[df[a] == 0]
    node.one = IGNode()
    node.zero = IGNode()
    tree_builder(df_one.drop(a, axis=1), node.one)
    tree_builder(df_zero.drop(a, axis=1), node.zero)




def create_tree(X, y):
    """

    :param X: matrix
    :param y: tagging
    :return:
    """
    df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)

    #df = df.iloc[[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, -1]]  # todo for testing
    status = df_leaf_status(df)

    if type(status) == int:
        return IGNode(label=status)

    a = get_max_IG_attr(df)

    root = IGNode(a)
    # now for each value, 0 or 1 that was in
    df_one = df.loc[df[a] == 1]
    df_zero = df.loc[df[a] == 0]
    root.one = IGNode()
    root.zero = IGNode()
    tree_builder(df_one.drop(a, axis=1), root.one)
    tree_builder(df_zero.drop(a, axis=1), root.zero)

    return root


def calc_IG(a, df):
    """
    calc entropy of attribute a of data set X
    :param: a column number
    :param: X the data frame
    :return: entropy
    """
    # calc for attribute a when all values are 0 how many are tagged 0 and how many are tagged 1

    total = len(df)
    #print(total)

    S_zero = df.loc[df[a] == 0]
    total_0 = len(S_zero)
    num_of_pos_0 = S_zero.iloc[:, -1].sum()
    #print('num of pos with atrr=0 is:', num_of_pos_0)
    num_of_neg_0 = total_0 - num_of_pos_0
    if num_of_pos_0 == 0 or num_of_neg_0 == 0:
        H0 = 0
    else:
        H0 = -(num_of_pos_0/total_0) * math.log(num_of_pos_0/total_0, 2) - (num_of_neg_0/total_0) * math.log(num_of_neg_0/total_0, 2)

    #print('HO is', H0)
    S_one = df.loc[df[a] == 1]
    num_of_pos_1 = S_one.iloc[:, -1].sum()
    #print('num of pos with atrr=1 is:', num_of_pos_1)
    total_1 = len(S_one)
    num_of_neg_1 = total_1 - num_of_pos_1
    if num_of_pos_1 == 0 or num_of_neg_1 == 0:
        H1 = 0
    else:
        H1 = -(num_of_pos_1/total_1) * math.log(num_of_pos_1/total_1, 2) - (num_of_neg_1/total_1) * math.log(num_of_neg_1/total_1, 2)

    #print(S_one)
    #print(S_zero)
    #print(total_0, total_1)

    entropy = (total_0 / total) * H0 + (total_1 / total) * H1
    return 1 - entropy


def get_max_IG_attr(df):
    best_col = 1
    max_IG = 0
    for col in df.columns[:-1]:  # for each col todo-----------maybe i need len+1
        cur_IG = calc_IG(col, df)
        if max_IG < cur_IG:
            max_IG = cur_IG
            best_col = col
    return best_col


def df_leaf_status(df):
    """

    :param df:
    :return: the label if all lables are equal, otherwise 'not a leaf'
    """
    s = df.iloc[:, -1].sum()
    if len(df.columns) == 1:
        # so there are no more attributes to split by
        if s > len(df) / 2: # so there are more positives than negatives
            return 1

        return 0

    if s == len(df):
        # so all rows are labeled positive, then its a leaf
        return 1
    if s == 0:
        # so all rows are labeled negative, then its a leaf
        return 0
    else:
        return 'not a leaf'


def predict_label(x, root):
    predictions = []
    for i, row in x.iterrows():
        cur = root
        while cur.label is None:
            cur_attr_val = x.iloc[i, cur.attribute]
            if x.iloc[i, cur.attribute] == 1:

                cur = cur.one
            else:
                cur = cur.zero

        predictions.append(cur.label)
    return predictions

### test
X, y = get_words_matrix()
#X = np.array([[1, 0, 1], [1, 0, 0], [0,0,1]], np.int32)
#y = np.array([1,1,0])
df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)

x = pd.DataFrame(X)

# create tree
root = create_tree(X, y)

# predict label
print('testing')
label = predict_label(x, root)
print('label is: ', label)
print()
print('tree is: ')
print(root)





