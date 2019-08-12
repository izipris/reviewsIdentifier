from learning.Tree import Tree


class RandomForest:
    def __init__(self, num_of_trees, Xy, tree_max_num_of_samples=None, tree_max_num_of_features=None):
        self.__Xy = Xy
        self.__num_of_trees = num_of_trees
        self.__tree_max_num_of_samples = tree_max_num_of_samples
        self.__tree_max_num_of_features = tree_max_num_of_features
        self.__trees = []

    def build(self):
        """Generates a random forest by the properties provided to the constructor."""
        for i in range(self.__num_of_trees):  # try on 300
            t = Tree(self.__Xy.astype(int), self.__tree_max_num_of_samples, self.__tree_max_num_of_features)
            t.build()
            self.__trees.append(t)
            print("Finished tree #" + str(i) + " with " + str(t.get_number_of_features()) + " features.")

    def predict(self, sample):
        """Returns a prediction for 'sample' - a words vector in the terms of the forest. 1-Positive, 0-Negative"""
        marked_yes = 0
        marked_no = 0
        for t in self.__trees:
            if t.predict(sample) == 1:
                marked_yes += 1
            else:
                marked_no += 1
        prediction = 1 if marked_yes >= marked_no else 0
        return prediction
