# Symbol Graph Genetic Programming
## 1. Introduction
Symbol Graph Genetic Programming (SGGP) is a novel symbolic regression approach that diverges from traditional GP methods by employing a semantic operator for the generation of new individuals. SGGP first initializes the population within the specially designed symbol graph (We employ the symbol graph to prove that the SR problem is NP-hard:[Paper]((https://github.com/SymbolGraph/sggp/blob/main/appendix/Symbolic%20Regression%20is%20NP-hard.pdf))). It then utilizes the semantic operator to generate new populations. The semantic operator employs the generalized Pareto distribution based on semantic similarity to assess the likelihood that each edge (subspace) in this graph will yield superior individuals. Guided by these probabilistic evaluations, SGGP strategically samples new individuals in its quest to discover accurate mathematical expressions. Comparative experiments conducted across three different benchmark types demonstrate that SGGP outperforms 21 existing baseline SR methods, achieving greater accuracy and conciseness in the mathematical expressions it generates.

## 2. Code
The codes are based on GPlearn's framework (https://github.com/trevorstephens/gplearn), where the computation of semantic similarity is in `SGGP._Program`, the estimation of generalized Pareto distribution is in `SGGP.genetic.BaseSymbolic`.
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
SGGP supports the following method to try out the regression task on your own data:
```
from SGGP.genetic import SymbolicRegressor

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
## 3. Experiment
We conducted a comprehensive evaluation of SGGP’s performance using three distinct types of benchmarks: [PMLB](https://epistasislab.github.io/pmlb/), [FSRB](https://space.mit.edu/home/tegmark/aifeynman.html), and [Strogatz](https://github.com/lacava/ode-strogatz). The results of SGGP are as follows. (For more detailed results, please refer to [Results](../appendix/SSGP_Results.pdf))
<img src="https://github.com/SymbolGraph/sggp/blob/main/appendix/Results%20on%20FSRB%20and%20Strogatz%20.png">
<img src="https://github.com/SymbolGraph/sggp/blob/main/appendix/Results%20on%20PMLB.png">

Compared with 14 symbolic regression methods and 7 other ML methods, the results clearly demonstrate that SGGP surpasses these baseline algorithms in terms of the $R^2$ Test, Model Size, and Solution Recovery Rate. 
