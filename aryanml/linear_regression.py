import numpy as np
class LinearRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weight=None
        self.bias=None
    def fit(self,X,Y): 
        n_samples,n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias
            
            #Transpose needed to multiply matrices
            dw=(-2/n_samples)*np.dot(X.T,(Y-(y_pred)))
            db=(-2/n_samples)*np.sum((Y-y_pred))
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self,X):
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred

