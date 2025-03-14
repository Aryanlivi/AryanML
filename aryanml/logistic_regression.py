import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.0001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weight=None
        self.bias=None
    
    def fit(self,X,Y):
        n_samples,n_features=X.shape
        self.weight = np.zeros((n_features, 1)) 
        self.bias=0
        for _ in range(self.n_iters):
            linear_model=np.dot(X,self.weight)+self.bias
            y_pred=self._sigmoid(linear_model)
            #Grad Descent Cost Function
            dw=(2/n_samples)*(np.dot(X.T,(y_pred-Y)))
            db=(2/n_samples)*(np.sum(y_pred - Y))
            self.weight-=(self.lr*dw)
            self.bias-=(self.lr*db)
            
    def classify(self,y_pred):
        # Apply threshold element-wise to the predictions
        y_pred_class = np.where(y_pred >= 0.5, 1, 0)
        return y_pred_class
    def predict(self,X):
        linear_model=np.dot(X,self.weight)+self.bias
        y_pred=self._sigmoid(linear_model)
        return self.classify(y_pred)
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
        