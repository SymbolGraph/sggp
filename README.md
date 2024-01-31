# Steiner Genetic Programming
## 1.Introduction
Steiner Genetic Programming (SteinerGP) is an innovative symbolic regression method. It begins by constructing a symbol graph to represent the mathematical expression space effectively. It then employs the generalized Pareto distribution based on semantic similarity to assess the likelihood that each edge (subspace) in this graph will yield superior individuals. Guided by these probabilistic evaluations, SteinerGP strategically samples new individuals in its quest to discover accurate mathematical expressions. Comparative experiments conducted across three different benchmark types demonstrate that SteinerGP outperforms 21 existing baseline SR methods, achieving greater accuracy and conciseness in the mathematical expressions it generates.

## 2. Code
Based on Trevor Stephens's framework (https://github.com/trevorstephens/gplearn), the core methods are in `SteinerGP.genetic.BaseSymbolic` and `SteinerGP._Program`.
### Requirements
Make sure you have installed the following Python version and packages before start running our code:
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
### Example
```
python exapmle.py
```
