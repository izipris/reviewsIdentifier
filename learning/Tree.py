import random
import numpy as np

class Tree:
    def __init__(self, Xy):
        """
        Tree constructor.
        Methodology: https://www.youtube.com/watch?v=7VeUPuFGJHk
        :param Xy: A data matrix where (m-1) first columns are features, and m column is tagging
        """
        num_of_samples = random.randint(3, int(np.ma.size(Xy, axis=0) / 2))
        # Select randomly samples
        random_pick_of_data = Xy[np.random.choice(Xy.shape[0], num_of_samples, replace=False), :]
        data_num_of_columns = np.ma.size(random_pick_of_data, axis=1)
        self.__data = random_pick_of_data[:, :data_num_of_columns - 1]
        self.__tagging = random_pick_of_data[:, data_num_of_columns - 1]
        # Select randomly features
        self.__features = random.sample(population=range(0, np.ma.size(self.__data, axis=1)),
                                        k=random.randint(3, int(np.ma.size(self.__data, axis=1) / 2)))