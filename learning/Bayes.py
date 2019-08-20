import math
import numpy as np
from scipy.stats import norm

PI = math.pi
NO_STD = 1/math.sqrt(2*PI)
TRAIN_RATIO = 0.85
PRUNE_ERR_RATIO = 0.3
ENOUGH_DATA = 600


# probability for single instance
def get_inst_prob(data, dist):
    return np.prod(norm.pdf(data, dist[:, 0], dist[:, -1]))


# probability for all instances. assumes isn't empty
def get_all_probs(data, dist):
    return np.apply_along_axis(get_inst_prob, 1, data, dist)


class Classifier:
    def __init__(self, Xy, attributes=None, prune=True, short_prune=False,
                 prune_err_thresh=0.49, prune_attr_thresh=200):
        print("Started learning process for Bayes Classifier.")
        self.stats = None
        self.attributes = None
        self.short_prune = short_prune
        self.prune_err_thresh = prune_err_thresh
        self.prune_attr_thresh = prune_attr_thresh
        self.build(Xy)  # get all stats
        if attributes is not None:
            self.attributes = np.array(attributes)
        if prune:
            self.prune_train(Xy, attributes)

    def prune_train(self, Xy, attributes):
        # naive prune - keep adding attributes in ascending order till it
        # stops improving classifier
        if attributes is None:  # then check all attributes
            attributes = np.array(range(Xy.shape[1] - 1)).astype(np.int)
        total_attr, total_data = attributes.size, Xy.shape[0]
        attr_errors = dict()  # remember each attr's lone-error
        # shorten pruning for very large data?
        if self.short_prune:
            print("Started SHORT feature-pruning.")
            print("Stage 1 of 2:")
            np.random.shuffle(attributes)  # random order of features
            good_enough = 0  # good-enough attributes counter
            idx = np.random.randint(total_data,
                                    size=min(int(PRUNE_ERR_RATIO *
                                                 total_data), ENOUGH_DATA))
            prune_data = Xy[idx]  # get random chunk of data to prune with
            for i in attributes:  # for each random attribute
                error = self.error(prune_data[:, :-1], prune_data[:, -1], [i])
                attr_errors[i] = error  # remember it's error
                if error < self.prune_err_thresh:
                    # keep count of times error is found below threshold
                    good_enough += 1
                    if good_enough%(self.prune_attr_thresh // 10) == 0:
                        print("\t", int((100*good_enough /
                              self.prune_attr_thresh) + 0.5), end="%\n")
                    if good_enough >= self.prune_attr_thresh:  # enough found
                        break
            print("Stage 2 of 2:")
            # sort attributes by error
            sorted_dict = sorted(attr_errors.items(), key=lambda x: x[1])
            # get best attr and it's error
            best = sorted_dict.pop(0)
            best_attr, min_error = best[0], best[1]
            best_attributes = [best_attr]  # remember it
            counter, dict_size = 0, len(sorted_dict) + 1
            while sorted_dict:  # while there are more attributes
                if counter % (dict_size // 10) == 0:
                    print("\t", int((100 * counter / dict_size) + 0.5),
                          end="%\n")
                counter += 1
                best = sorted_dict.pop(0)  # get best
                best_attr = best[0]
                # check it's combined error with chosen best-attributes
                error = self.error(prune_data[:, :-1], prune_data[:, -1],
                                   np.concatenate((best_attributes,
                                                   [best_attr])))
                if error < min_error:  # if it helps - keep it
                    min_error = error
                    # print('added ', best_attr, "error: ", error)
                    best_attributes.append(best_attr)
        else:
            print("Started FULL feature-pruning.")
            print("Stage 1 of 2:")
            for i, att in enumerate(attributes):
                error = self.error(Xy[:, :-1], Xy[:, -1], [att])
                attr_errors[att] = error
                if i%(total_attr//10) == 0:
                    print("\t", int((100*i/total_attr) + 0.5), end="%\n")
            print("Stage 2 of 2:")
            # sort attributes by error
            sorted_dict = sorted(attr_errors.items(), key=lambda x: x[1])
            # get best attr and it's error
            best = sorted_dict.pop(0)
            best_attr, min_error = best[0], best[1]
            best_attributes = [best_attr]  # remember it
            counter, dict_size = 0, len(sorted_dict) + 1
            while sorted_dict:  # while there are more attributes
                if counter%(dict_size//10) == 0:
                    print("\t", int((100*counter/dict_size) + 0.5), end="%\n")
                counter += 1
                best = sorted_dict.pop(0)  # get best
                best_attr = best[0]
                # check it's combined error with chosen best-attributes
                error = self.error(Xy[:, :-1], Xy[:, -1],
                                   np.concatenate((best_attributes, [best_attr])))
                if error < min_error:  # if it helps - keep it
                    min_error = error
                    # print('added ', best_attr, "error: ", error)
                    best_attributes.append(best_attr)
        print("Selected", len(best_attributes), "attributes out of",
              total_attr, "total.")
        self.attributes = np.array(best_attributes)

    def build(self, Xy):
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

    def predict(self, data, attributes=None):
        # get probabilities for each label, return label with better prob
        # for each instance
        data = np.array(data)
        if data.size == 0:
            return 0.5  # as in random
        if len(data.shape) <= 1:  # if just one instance then array --> matrix
            data = np.array([data])
        zero_stats = self.stats[0]
        one_stats = self.stats[1]
        if self.attributes is not None:  # if have list of relevant attributes
            data = data[:, self.attributes]
            zero_stats = zero_stats[self.attributes]
            one_stats = one_stats[self.attributes]
        elif attributes is not None:  # if given specific attributes to
            # focus on
            data = data[:, attributes]
            zero_stats = zero_stats[attributes]
            one_stats = one_stats[attributes]
        zero_probabilities = get_all_probs(data, zero_stats)
        one_probabilities = get_all_probs(data, one_stats)
        prediction = np.argmax([zero_probabilities, one_probabilities], axis=0)
        return prediction

    def error(self, data, labels, attributes=None):
        data = np.array(data)
        if len(data.shape) <= 1:
            return int(self.predict(data, attributes) != labels)
        error = np.sum(labels != self.predict(data, attributes)) / float(
            len(labels))
        return error
