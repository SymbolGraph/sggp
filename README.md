# Steiner Genetic Programming
## 1.Introduction
Steiner Genetic Programming (SteinerGP) is a novel symbolic regression approach that diverges from traditional GP methods by employing a semantic operator for the generation of new individuals. SteinerGP first initializes the population within the specially designed symbol graph. It then utilizes the semantic operator to generate new populations. The semantic operator employs the generalized Pareto distribution based on semantic similarity to assess the likelihood that each edge (subspace) in this graph will yield superior individuals. Guided by these probabilistic evaluations, SteinerGP strategically samples new individuals in its quest to discover accurate mathematical expressions. Comparative experiments conducted across three different benchmark types demonstrate that SteinerGP outperforms 21 existing baseline SR methods, achieving greater accuracy and conciseness in the mathematical expressions it generates.

## 2. Code
The codes are based on GPlearn's framework (https://github.com/trevorstephens/gplearn), where the computation of semantic similarity is in `SteinerGP._Program`, the estimation of generalized Pareto distribution is in `SteinerGP.genetic.BaseSymbolic`.
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
### Getting started
SteinerGP supports the following method to try out the regression task on your own data:
```
from SteinerGP.genetic import SymbolicRegressor

rng = check_random_state(0)
# Create some data
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0]**2 - np.sin(X_train[:, 1]) + X_train[:, 1] - 1

# Create the model
est_gp = SymbolicRegressor()

# Fit the model
est_gp.fit(X_train, y_train)

# View the best expression
print(est_gp._program)
```
