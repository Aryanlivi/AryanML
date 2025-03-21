import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    def predict(self,X):
        y_pred=[self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,X):
        #compute distances:
        distances=[euclidean_distance(X,X_train) for X_train in self.X_train]
        
        #get K nearest samples,
        k_indices=np.argsort(distances)[:self.k]
        k_neighbor_labels=[self.y_train[i] for i in k_indices]
        most_common_label=Counter(k_neighbor_labels).most_common(1)
        return most_common_label[0][0]