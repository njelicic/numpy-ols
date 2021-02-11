# numpy-ols
Ordinary Least Squares in just NumPy. Follows the Sklearn API. Results should be the same. 

# Usage:

```
from ols import OLS

clf = OLS(fit_intercept=True)

clf.fit(X,y)

clf.predict(X_test)

clf.summary(feature_names)              #if fit_intercept = False
clf.summary(feature_names + ['const'])  #if fit_intercept = True

````

