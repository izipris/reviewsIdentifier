# python file responsible for creating a classification tree using the ID3 algorithm
import pandas as pd
from learning.TreeNode import CustomTreeNode
from testing import get_words_matrix
import math


def create_tree(X, y):
    """

    :param X: matrix
    :param y: tagging
    :return:
    """
    root = CustomTreeNode(matrix=None)


def calc_IG(a, X, y):
    """
    calc entropy of attribute a of data set X
    :param: a column number
    :param: X the data frame
    :return: entropy
    """
    # calc for attribute a when all values are 0 how many are tagged 0 and how many are tagged 1


    df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
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


def get_max_IG_attr(X, y):
    best_col = 1
    max_IG = 0
    for i in range(1, len(X[0])):  # for each col todo-----------maybe i need len+1
        cur_IG = calc_IG(i, X, y)
        if max_IG < cur_IG:
            max_IG = cur_IG
            best_col = i
    return best_col



### test
X, y = get_words_matrix()
attr = get_max_IG_attr(X, y)
print(attr)