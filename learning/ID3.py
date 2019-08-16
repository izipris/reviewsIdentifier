# python file responsible for creating a classification tree using the ID3 algorithm
import pandas as pd
import math


class IGNode:
    def __init__(self, attribute=None, label=None, one=None, zero=None, parent=None):
        self.attribute = attribute  # the attribute this node splits by
        self.label = label  # in case this node is a leaf it has no attribute but only a label
        self.one = one  # subtree for value is one
        self.zero = zero  # subtree for value is zero
        self.parent = parent

    def __str__(self):
        h = self.get_height()
        if h == 0:
            # its a leaf
            return 'leaf label: ' + str(self.label)
        return str(self.attribute) + '--->' + str(self.one) + '|||' + str(self.zero)

    def display(self):
        lines, _, _, _ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.one is None and self.zero is None:
            line = '%s' % self.label
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.one is None:
            lines, n, p, x = self.zero._display_aux()
            s = '%s' % self.attribute
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.zero is None:
            lines, n, p, x = self.one._display_aux()
            s = '%s' % self.attribute
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.zero._display_aux()
        right, m, q, y = self.one._display_aux()
        s = '%s' % self.attribute
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

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

    def __init__(self, Xy, training_fraction=0.5, max_depth=math.inf):
        """

        :param Xy: all data in matrix
        :param training_fraction: fraction of samples to use for training
        :param max_depth: max depth IG tree should go
        """

        Xy = pd.DataFrame(Xy)
        self.root = None  # root of IG tree
        self.max_depth = max_depth

        train = Xy.sample(frac=training_fraction, random_state=150).reset_index(drop=True)  # randaom_state=200
        self.Xy = train  # training set

        test_Xy = Xy.drop(train.index).reset_index(drop=True)
        self.test_set = test_Xy.iloc[:, :-1]  # hold out set no labels
        self.true_y = test_Xy.iloc[:, -1].tolist()  # true labels of holdout set

    def train(self):
        """
        train method for classifier
        :return:
        """
        print('beginning training')
        self.create_tree()
        print('ended training')

    def predict(self, x=None):
        """
        method to predict the test data, or x if given as input
        :param x: a data frame to predict, if not specified, our own holdout set will be used
        :return:
        """
        if x is None:
            x = self.test_set

        if self.root is None:
            print('must train first!')
            return

        predictions = []
        for i, row in x.iterrows():
            cur = self.root
            while cur.label is None:  # traverse tree by where it points me, until we reach our leaf
                if x.iloc[i, cur.attribute] == 1:
                    cur = cur.one
                else:
                    cur = cur.zero

            predictions.append(cur.label)  # prediction is label of leaf we've reached
        return predictions

    def prune(self):
        """
        prunes tree according to hold out data
        :return:
        """
        # run dfs on tree and find leaves
        #print('-------')
        #print('tree before pruning: ')
        #self.root.display()
        self.dfs(self.root.one)
        self.dfs(self.root.zero)
        #print('after pruning: ')
        #self.root.display()
        #print('------')

    def dfs(self, cur):
        if cur.zero is None and cur.one is None: # its a leaf
            self.prune_leaf(cur)

        if cur.zero is not None:
            self.dfs(cur.zero)

        if cur.one is not None:
            self.dfs(cur.one)

    def prune_leaf(self, cur):

        # calc error while leaf is here
        #print('------prune leaf-----')
        #print('error before pruning this leaf: ')
        e1 = self.check_hold_out_error()
        # remove leaf => replace parent with label that is most examples from training set

        s = self.Xy.iloc[:, -1].sum()
        if s > len(self.Xy) / 2:  # so there are more positives than negatives
            cur.parent.label = 1

        else:
            cur.parent.label = 0

        r0 = cur.parent.zero
        r1 = cur.parent.one
        cur.parent.zero = None
        cur.parent.one = None
        #print('**** this is how tree looks if i were to prune ****')
        #self.root.display()
        #print('***************************************************')

        # calc error whithout leaf
        #print('error after pruning: ')
        e2 = self.check_hold_out_error()

        # if error is not better now, bring back leaf
        if e1 < e2:
            #print('was better before i pruned, going back')
            # restore tree how it was
            cur.parent.zero = r0
            cur.parent.one = r1
            cur.parent.label = None
        else:
            print('succesful prune!')
        #    self.root.display()
        #print('------end prune leaf--------')

    @staticmethod
    def get_attributes_from_tree(cur):
        if cur.label is not None:  # its a leaf
            return []
        lst = [cur.attribute]
        if cur.zero is not None:
            lst += IGClassifier.get_attributes_from_tree(cur.zero)
        if cur.one is not None:
            lst += IGClassifier.get_attributes_from_tree(cur.one)
        return lst






    def check_hold_out_error(self):
        """
        method for checking error rate once this classifier is trained
        error rate is checked according to our stored holdout data that we kept on the side
        :return:
        """
        # check error:
        label = self.predict()
        error = 0
        for i in range(len(label)):
            if int(label[i]) != int(self.true_y[i]):
                error += 1
        print('hold out error rate is: ', error / len(label))
        return error/ len(label)

    def check_train_error(self):
        """
        method for checking error rate once this classifier is trained
        error rate is checked on our trained data
        :return:
        """
        # check error:
        label = self.predict(self.Xy.iloc[:, :-1])  # use train set without labels
        error = 0
        for i in range(len(label)):
            if int(label[i]) != int(self.true_y[i]):
                error += 1
        print('training error rate is: ', error / len(label))
        return error / len(label)

    def create_tree(self):
        """
        create tree for our data
        :return:
        """
        self.Xy.iloc[:, -1] = self.Xy.iloc[:, -1].astype(int) # make labels an int to add and subtract with them

        status = IGClassifier.df_leaf_status(self.Xy)

        if type(status) == int:  # so root of tree is a leaf
            return IGNode(label=status)

        a = IGClassifier.get_max_IG_attr(self.Xy)  # gets best attribute to split by through IG

        root = IGNode(a)  # create first split of tree
        # now split our data in to 2, one for each possible value of attribute
        df_one = self.Xy.loc[self.Xy[a] == 1]
        df_zero = self.Xy.loc[self.Xy[a] == 0]
        # create kids
        root.one = IGNode(parent=root)
        root.zero = IGNode(parent=root)
        # call recursive function to build each child while dropping attribute a
        self.tree_builder(df_one.drop(a, axis=1), root.one, 1)
        self.tree_builder(df_zero.drop(a, axis=1), root.zero, 1)

        self.root = root

    def tree_builder(self, cur_Xy, node, depth):
        """recursive helper function similar to create tree """
        status = IGClassifier.df_leaf_status(cur_Xy)

        # depth control
        if depth > self.max_depth:
            s = cur_Xy.iloc[:, -1].sum()
            if s > len(cur_Xy) / 2:  # so there are more positives than negatives
                node.label = 1
                return
            node.label = 0
            return

        if type(status) == int:
            node.label = status
            return

        a = IGClassifier.get_max_IG_attr(cur_Xy)
        node.attribute = a
        df_one = cur_Xy.loc[cur_Xy[a] == 1]
        df_zero = cur_Xy.loc[cur_Xy[a] == 0]
        node.one = IGNode(parent=node)
        node.zero = IGNode(parent=node)
        self.tree_builder(df_one.drop(a, axis=1), node.one, depth+1)
        self.tree_builder(df_zero.drop(a, axis=1), node.zero, depth+1)

    @staticmethod
    def get_max_IG_attr(Xy):
        """
        get attribute with best IG from Xy
        :param Xy:
        :return:
        """
        best_col = 0  # bug prone!!! todo - some key error is happening becuase this was bestcol = 1
        max_IG = 0
        for col in Xy.columns[:-1]:  # for each col
            cur_IG = IGClassifier.calc_IG(col, Xy)
            if max_IG < cur_IG:
                max_IG = cur_IG
                best_col = col
        return best_col

    @staticmethod
    def calc_IG(a, Xy):
        """
        calc entropy of attribute a of data set Xy
        :param: a column number
        :param: Xy the data frame
        :return: IG
        """
        # calc for attribute a when all values are 0 how many are tagged 0 and how many are tagged 1

        total = len(Xy)

        S_zero = Xy.loc[Xy[a] == 0]
        total_0 = len(S_zero)
        num_of_pos_0 = S_zero.iloc[:, -1].sum()

        num_of_neg_0 = total_0 - num_of_pos_0
        if num_of_pos_0 == 0 or num_of_neg_0 == 0:
            H0 = 0
        else:
            H0 = -(num_of_pos_0/total_0) * math.log(num_of_pos_0/total_0, 2) - (num_of_neg_0/total_0) * math.log(num_of_neg_0/total_0, 2)

        S_one = Xy.loc[Xy[a] == 1]
        num_of_pos_1 = S_one.iloc[:, -1].sum()
        total_1 = len(S_one)

        num_of_neg_1 = total_1 - num_of_pos_1
        if num_of_pos_1 == 0 or num_of_neg_1 == 0:
            H1 = 0
        else:
            H1 = -(num_of_pos_1/total_1) * math.log(num_of_pos_1/total_1, 2) - (num_of_neg_1/total_1) * math.log(num_of_neg_1/total_1, 2)

        entropy = (total_0 / total) * H0 + (total_1 / total) * H1
        return 1 - entropy

    @staticmethod
    def calc_ratio(a, Xy):
        """
        calc ratio of split on a, not IG
        best ratio is closest to 0 or 1
        todo- not in use as of now is not operational
        :param a:
        :param Xy:
        :return:
        """
        total = len(Xy)

        S_zero = Xy.loc[Xy[a] == 0]
        total_0 = len(S_zero)
        num_of_pos_0 = S_zero.iloc[:, -1].sum()
        num_of_neg_0 = total_0 - num_of_pos_0

        S_one = Xy.loc[Xy[a] == 1]
        total_1 = len(S_one)
        num_of_pos_1 = S_one.iloc[:, -1].sum()
        num_of_neg_1 = total_1 - num_of_pos_1


    @staticmethod
    def df_leaf_status(Xy):
        """
        :param Xy: data to get leaf of
        :return: the label if all labels are equal, otherwise 'not a leaf'
        """
        s = Xy.iloc[:, -1].sum()
        if len(Xy.columns) == 1:
            print('reached max depth, creating leaf')
            # so there are no more attributes to split by
            if s > len(Xy) / 2:  # so there are more positives than negatives
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
