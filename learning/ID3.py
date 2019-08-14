# python file responsible for creating a classification tree using the ID3 algorithm
import pandas as pd
import numpy as np

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

        return str(self.attribute) + '--->' + str(self.one) + '|||' +  str(self.zero)

    def get_height(self):
        h = 0
        if self.one is not None:
            h = self.one.get_height() + 1

        if self.zero is not None:
            h_temp = self.zero.get_height() + 1
            if h_temp > h:
                h = h_temp

        return h


class IGClassifier:

    def __init__(self, Xy, training_fraction=0.5):
        print('initializing IGClassifier')
        ####important####
        # X, y = get_words_matrix()
        # df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)
        ### im assuming what is df up here us what Xy comes out to be down there
        Xy = pd.DataFrame(Xy)
        self.root = None

        train = Xy.sample(frac=training_fraction, random_state=200).reset_index(drop=True)
        self.Xy = train

        test_Xy = Xy.drop(train.index).reset_index(drop=True)
        self.test_set = test_Xy.iloc[:, :-1]
        self.true_y = test_Xy.iloc[:, -1].tolist()

    def train(self):
        print('beginning training')
        self.create_tree()
        print('ended training')

    def predict(self, x=None):
        if not x:
            x = self.test_set

        if self.root is None:
            print('must train first!')
            return

        predictions = []
        for i, row in x.iterrows():
            cur = self.root
            while cur.label is None:
                cur_attr_val = x.iloc[i, cur.attribute]
                if x.iloc[i, cur.attribute] == 1:

                    cur = cur.one
                else:
                    cur = cur.zero

            predictions.append(cur.label)
        return predictions

    def check_error(self):
        print('checking error rate')
        # check error:
        label = self.predict()
        error = 0
        for i in range(len(label)):
            if int(label[i]) != int(self.true_y[i]):
                error += 1
        print('error rate is: ', error / len(label))

    def create_tree(self):
        """

        :return:
        """
        self.Xy.iloc[:, -1] = self.Xy.iloc[:, -1].astype(int) # make lables an int to add and subtract with them

        status = IGClassifier.df_leaf_status(self.Xy)

        if type(status) == int:
            return IGNode(label=status)

        a = IGClassifier.get_max_IG_attr(self.Xy)

        root = IGNode(a)
        # now for each value, 0 or 1 that was in
        df_one = self.Xy.loc[self.Xy[a] == 1]
        df_zero = self.Xy.loc[self.Xy[a] == 0]
        root.one = IGNode()
        root.zero = IGNode()
        IGClassifier.tree_builder(df_one.drop(a, axis=1), root.one)
        IGClassifier.tree_builder(df_zero.drop(a, axis=1), root.zero)

        self.root = root

    @staticmethod
    def tree_builder(cur_Xy, node):
        """recursive function"""
        status = IGClassifier.df_leaf_status(cur_Xy)

        if type(status) == int:
            node.label = status
            return

        a = IGClassifier.get_max_IG_attr(cur_Xy)
        # now for each value, 0 or 1 that was in
        node.attribute = a
        df_one = cur_Xy.loc[cur_Xy[a] == 1]
        df_zero = cur_Xy.loc[cur_Xy[a] == 0]
        node.one = IGNode()
        node.zero = IGNode()
        IGClassifier.tree_builder(df_one.drop(a, axis=1), node.one)
        IGClassifier.tree_builder(df_zero.drop(a, axis=1), node.zero)

    @staticmethod
    def calc_IG(a, Xy):
        """
        calc entropy of attribute a of data set X
        :param: a column number
        :param: X the data frame
        :return: entropy
        """
        # calc for attribute a when all values are 0 how many are tagged 0 and how many are tagged 1

        total = len(Xy)
        #print(total)

        S_zero = Xy.loc[Xy[a] == 0]
        total_0 = len(S_zero)
        num_of_pos_0 = S_zero.iloc[:, -1].sum()
        #print('num of pos with atrr=0 is:', num_of_pos_0)
        num_of_neg_0 = total_0 - num_of_pos_0
        if num_of_pos_0 == 0 or num_of_neg_0 == 0:
            H0 = 0
        else:
            H0 = -(num_of_pos_0/total_0) * math.log(num_of_pos_0/total_0, 2) - (num_of_neg_0/total_0) * math.log(num_of_neg_0/total_0, 2)

        #print('HO is', H0)
        S_one = Xy.loc[Xy[a] == 1]
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

    @staticmethod
    def get_max_IG_attr(Xy):
        best_col = 1
        max_IG = 0
        for col in Xy.columns[:-1]:  # for each col todo-----------maybe i need len+1
            cur_IG = IGClassifier.calc_IG(col, Xy)
            if max_IG < cur_IG:
                max_IG = cur_IG
                best_col = col
        return best_col

    @staticmethod
    def df_leaf_status(Xy):
        """

        :param df:
        :return: the label if all lables are equal, otherwise 'not a leaf'
        """
        s = Xy.iloc[:, -1].sum()
        if len(Xy.columns) == 1:
            print('reached max depth, creating leaf')
            # so there are no more attributes to split by
            if s > len(Xy) / 2: # so there are more positives than negatives
                return 1

            return 0

        if s == len(Xy):
            # so all rows are labeled positive, then its a leaf
            return 1
        if s == 0:
            # so all rows are labeled negative, then its a leaf
            return 0
        else:
            return 'not a leaf'
