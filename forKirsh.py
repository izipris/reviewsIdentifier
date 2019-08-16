from utilities.DataUtils import DataUtils
import numpy as np
import pandas as pd
from learning.DataHolder import DataHolder
from learning.ID3 import IGClassifier
from test_IG import get_model, calc_error

model = get_model(new=False, filename='smallTest.txt')  ######## - choose data file here
X = model[0][0]
y = model[0][1]
Xy = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, ignore_index=True)

#new version -- not in use now makes too many attributes for this tiny data set
#model = get_model(new=True, filename='smallTest.txt')
#Xy = pd.DataFrame(model[0])

train_Xy = Xy.sample(frac=0.7, random_state=109).reset_index(drop=True)  # split of train to test data
test_Xy = Xy.drop(train_Xy.index).reset_index(drop=True)
test_set = test_Xy.iloc[:, :-1]  # test set, no labels
true_y = test_Xy.iloc[:, -1].tolist()  # true labels of test set

words_matrix = train_Xy

# here choose your max depth of the tree, if not specified will be math.inf, this will let u handle the run time
ig_tree = IGClassifier(words_matrix, max_depth=15, training_fraction=0.5)  # rest of fraction used to prune
ig_tree.train()

ig_tree.root.display()  # this is to vizualize the tree




### if u wanna calc the error from the tree:
# before pruning:
label = ig_tree.predict(test_set)
# calc error
labels = ig_tree.predict(test_set)
print('test error is: ', calc_error(labels, true_y))

#pruning
ig_tree.prune()

labels = ig_tree.predict(test_set)
print('after prune test error is: ', calc_error(labels, true_y))
print('pruned tree is:')
ig_tree.root.display()


# what u wanted:
attributes = IGClassifier.get_attributes_from_tree(ig_tree.root)