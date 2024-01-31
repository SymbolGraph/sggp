# Steiner Genetic Programming
## 1.Introduction
Steiner Genetic Programming (SteinerGP) is a novel symbolic regression approach that diverges from traditional GP methods by employing a semantic operator for the generation of new individuals. SteinerGP first initializes the population within the specially designed symbol graph. It then utilizes the semantic operator to generate new populations. The semantic operator employs the generalized Pareto distribution based on semantic similarity to assess the likelihood that each edge (subspace) in this graph will yield superior individuals. Guided by these probabilistic evaluations, SteinerGP strategically samples new individuals in its quest to discover accurate mathematical expressions. Comparative experiments conducted across three different benchmark types demonstrate that SteinerGP outperforms 21 existing baseline SR methods, achieving greater accuracy and conciseness in the mathematical expressions it generates.

## 2. Code
The codes are based on Trevor Stephens's framework (https://github.com/trevorstephens/gplearn), where the computation of semantic similarity is in `SteinerGP._Program`, and the estimation of generalized Pareto distribution is in `SteinerGP.genetic.BaseSymbolic`.
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
