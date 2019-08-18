import math
import numpy as np
from scipy.stats import norm

PI = math.pi

NO_STD = 1/math.sqrt(2*PI)


# probability for single instance
def get_inst_prob(data, dist):
    return np.prod(norm.pdf(data, dist[:, 0], dist[:, -1]))


# probability for all instances. assumes isn't empty
def get_all_probs(data, dist):
    if data.ndim == 1:
        return get_inst_prob(data, dist)
    return np.apply_along_axis(get_inst_prob, 1, data, dist)


class Classifier:
    def __init__(self, Xy):
        # separate instances by label
        zeros_inst = Xy[np.where(Xy[:, -1] == 0)][:, 0:-1]
        ones_inst = Xy[np.where(Xy[:, -1] == 1)][:, 0:-1]
        # get means. switch nan to 0.5
        zeros_mean = np.mean(zeros_inst, axis=0)
        zeros_mean[np.isnan(zeros_mean)] = 0.5
        ones_mean = np.mean(ones_inst, axis=0)
        ones_mean[np.isnan(ones_mean)] = 0.5
        # get standard deviations. switch nan, 0 to 1/sqrt(2*pi)
        zeros_std = np.std(zeros_inst, axis=0)
        zeros_std[np.isnan(zeros_std)] = NO_STD
        zeros_std[np.where(zeros_std == 0)] = NO_STD
        ones_std = np.std(ones_inst, axis=0)
        ones_std[np.isnan(ones_std)] = NO_STD
        ones_std[np.where(ones_std == 0)] = NO_STD
        # combine each labels stats
        zero_stats = np.concatenate(([zeros_mean], [zeros_std])).transpose()
        one_stats = np.concatenate(([ones_mean], [ones_std])).transpose()
        # make library
        self.stats = (zero_stats, one_stats)
        # print(self.stats)  # todo remove

    def predict(self, data):
        # get probabilities for each label, return label with better prob
        # for each instance
        data = np.array(data)
        if data.size == 0:
            return 0.5  # as in random
        zero_probabilities = get_all_probs(data, self.stats[0])
        # zero_probabilities[np.isnan(zero_probabilities)] = 0
        one_probabilities = get_all_probs(data, self.stats[1])
        # one_probabilities[np.isnan(one_probabilities)] = 0
        prediction = np.argmax([zero_probabilities, one_probabilities], axis=0)
        # print('prediction: ', prediction)  # todo remove
        return prediction

    def error(self, data, labels):
        data = np.array(data)
        if data.ndim <= 1:
            return int(self.predict(data) != labels)
        error = np.sum(labels != self.predict(data)) / float(len(labels))
        return error


# --------------concat test
# a = np.array([[1, 2],
#               [3, 4],
#               [5, 6],
#               [7, 8]])
# b = np.array([9, 10, 11, 12])
# print(np.concatenate((a, np.array([b]).T), axis=1))

# -------------apply along axis test
# def my_test(data):
#     print(scpy.norm(data[0], data[1]).pdf(data[2]))
#
#
# a = np.array([[73, 6.2, 71.5],
#               [1, 0.5, 1.1],
#               [20, 5, 1.1]])
# np.apply_along_axis(my_test, 1, a)

# -------------replace nan with 0 or 1 test
# a = np.array([1, 0, np.nan, 5, np.nan])
# print('reg: ', a)
# print('zeros: ', np.nan_to_num(a))
# a[np.isnan(a)] = 1
# print('ones: ', a)


# # -------------probs and predict test
# data = np.array([[1, 1, 1, 0], [0, 0.5, 0.5, 1]])
# dist1 = np.array([[1, 0],
#                  [0.5, 0.5],
#                  [0.5, 0.5],
#                  [0, 0]])
# dist0 = np.array([[0, 0],
#                  [0.5, 0.5],
#                  [0.5, 0.5],
#                  [1, 0]])
# dist1[np.where(dist1[:, -1] == 0), -1] = NO_STD
# dist0[np.where(dist0[:, -1] == 0), -1] = NO_STD
# probs0 = get_all_probs(data, dist0)
# probs1 = get_all_probs(data, dist1)
# print('dist1:\n', dist1)
# print('dist0:\n', dist0)
# print('probs 1:\n', probs1)
# print('probs 0:\n', probs0)
# prediction = np.argmax([probs0, probs1], axis=0)
# print('prediction is:\n', prediction)


###

# # ---------this is garbage
# # for single attribute
# def get_prob(data):
#     prob = norm(data[0], data[1]).pdf(data[2])
#     return prob
#
#
# # for single instance
# def concat_and_prob(data, dist):
#     concat = np.concatenate((dist, np.array([data]).T), axis=1)
#     return np.apply_along_axis(get_prob, 1, concat)
