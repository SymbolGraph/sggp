## 1. Code
### Requirements
Make sure you have installed the following Python packages before start running our code:
* scikit-learn 
* numpy
* numbers
* copy
* scipy
* joblib
* abc
* warnings
* time
* random
* itertools
* torch
### Getting started
SGFCNN supports the following method to try out the regression task on your own data:
```
from SGFCNN.genetic import SymbolicRegressor
from pmlb.pmlb import fetch_data
from sklearn.model_selection import train_test_split

filename="feynman_II_37_1"
# Fetch the dataset
X, y = fetch_data(filename, return_X_y=True, local_cache_dir="/Users/songjinglu/Desktop/datasets")

# # Split the data into training and testing sets
split_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)
y_real=[]

for i in range(10):
    est_gp = SymbolicRegressor(population_size=1000,
                               generations=1000, stopping_criteria=1e-5,
                               verbose=1,
                               random_state=None,n_features=X_train.shape[1])
    est_gp.fit(X_train, y_train,y_real=y_real,parts=filename,xnum=i)
```
