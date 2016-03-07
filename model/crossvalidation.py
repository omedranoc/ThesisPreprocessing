from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

X = np.array([[1, 2], [4, 4], [1, 7], [3, 5]])
y = np.array([0, 0, 1, 1])
sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
len(sss)
print(sss)       

for train_index, test_index in sss:
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]


