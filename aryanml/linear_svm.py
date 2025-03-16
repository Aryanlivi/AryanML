import numpy as np

class LinearSVM:
    def __init__(self,lr=0.001,lamba_param=0.01,n_iters=1000):
        self.lr=lr
        self.lamba_param=lamba_param
        self.n_iters=n_iters
        self.weight=None
        self.bias=None
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        yi=np.where(y<=0,-1,1)
        for _ in range(self.n_iters):
            for idx,xi in enumerate(X):
                fx=np.dot(xi,self.weight)-self.bias
                condition=yi[idx]*fx
                if(condition>=1):
                    self.weight-=self.lr*(2*self.lamba_param*self.weight)
                else:
                    self.weight-=self.lr*((2*self.lamba_param*self.weight)-np.dot(xi,yi[idx]))
                    self.bias-=self.lr*yi[idx]
        
    def predict(self,X):
        approx = np.dot(X, self.weight) - self.bias  
        return np.sign(approx) #return class -1,0 or 1.
    