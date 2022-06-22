import numpy as np

class OLS():    
    """"Custom Ordinary Least Squares Class"""    
    def __init__(self,fit_intercept=True):
        self.fit_intercept = fit_intercept
        return
    
    def fit(self,X,y):
        #find least squares solution and store the weights in W
        self.X = X
        self.y = y
        
        if self.fit_intercept:
            X = self.add_intercept(X)
            
        self.W = np.linalg.lstsq(X,y,rcond=None)[0]
        return 
    
    def add_intercept(self,X):
        a = np.ones((X.shape[0],1))
        return np.concatenate((X,a),axis=1) 
    
    def predict(self,X):
        # matrix multiply X with the transposed weights
        
        if self.fit_intercept:
            X = self.add_intercept(X)
        return X @ self.W.T    
    
    def score(self):
        #calculate the root mean squared error
        y_hat = self.predict(self.X)
        print(f'\nRoot mean squared error: {np.sqrt(np.mean(((self.y-y_hat)**2)))}')
        return
    
    def summary(self,names):
        print('OLS Summary\n\nVariable\tCoefficient')
        
        if self.fit_intercept:
            names.append('const')
        
        for n,w in zip(names, self.W):
            print(f'{n}\t{w}' )
        self.score()
        return
