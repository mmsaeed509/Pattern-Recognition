"""
######################################
#
# 﫥  @author   : 00xWolf
#   GitHub    : @mmsaeed509
#   Developer : Mahmoud Mohamed
#
######################################
"""

from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import COLORS


# calculate distance between x, y (use to get distance between input and dataset in line No. 29) #
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN:
    def __init__(self, k=3):
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # detect which class that input belongs to #
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance between input and dataset #
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # sort & get the closest k #
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote (most common) #
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()  # load dataset #
X, y = iris.data, iris.target  # split to x, y  #

# split train, test                                       20% for test #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

clf = KNN(k=5)
clf.fit(X_train, y_train)  # train #
predictions = clf.predict(X_test)  # test (predict) #

print(COLORS.BOLD_HIGH_INTENSITY_RED + "[*] Printing Info.")
print(COLORS.BOLD_HIGH_INTENSITY_CYAN + "[+] The class that input belongs to: ")
print(predictions)  # print class that input belongs to #

acc = np.sum(predictions == y_test) / len(y_test)  # calculate accuracy #
print(COLORS.BOLD_HIGH_INTENSITY_PURPLE + "[+] Knn From scratch accuracy: " + str(acc))  # print accuracy percentage #

# compare #
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)  # train #
predictions = neigh.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test)  # calculate accuracy #
print(COLORS.BOLD_HIGH_INTENSITY_PURPLE + "[+] Built-in accuracy: " + str(acc))  # print accuracy percentage #

print(COLORS.BOLD_HIGH_INTENSITY_GREEN + "[✔] D O N E!")

print(COLORS.BOLD_HIGH_INTENSITY_RED + "[✘] Closing" + COLORS.RESET_COLOR)
