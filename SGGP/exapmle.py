from SGGP.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
import numpy as np

x0 = np.arange(-1, 1, .1)
x1 = np.arange(-1, 1, .1)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - np.sin(x1) + x1 - 1

rng = check_random_state(0)

# Training samples
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0]**2 - np.sin(X_train[:, 1]) + X_train[:, 1] - 1

# Testing samples
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
y_test = X_test[:, 0]**2 - np.sin(X_test[:, 1]) + X_test[:, 1] - 1

est_gp = SymbolicRegressor(population_size=10,generations=1000, stopping_criteria=1e-5,
                        verbose=1, random_state=None,n_features=X_train.shape[1])
est_gp.fit(X_train, y_train)