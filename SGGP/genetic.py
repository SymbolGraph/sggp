


import itertools
from abc import ABCMeta, abstractmethod
import random
from time import time
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from scipy.stats import genpareto

from ._program import _Program
from .fitness import _fitness_map, _Fitness
from .functions import _function_map, _Function, sig1 as sigmoid
from .utils import _partition_estimators
from .utils import check_random_state

__all__ = ['SymbolicRegressor']

MAX_INT = np.iinfo(np.int32).max


def shape(X, n):
    X_sorted = np.sort(X)
    sum_xi = 0
    for i in range(n):
        sum_xi += X_sorted[i]
    sum_eta = 1e-5
    for i in range(n):
        for j in range(i + 1, n):
            sum_eta += X_sorted[j] - X_sorted[i]
    return 2 - (n - 1) * sum_xi / sum_eta, sum_xi

def calculate_extreme(array, extreme_thre):
    total_sum = []
    n = 0
    for i in range(len(array)):
        for j in range(len(array[i]) - 6):
            total_sum.append(array[i][j])
            n += 1
    if n != 0:
        extreme = np.percentile(total_sum, extreme_thre)
    else:
        extreme = 0
    return extreme

def _parallel_evolve(n_programs, parents, X, y, sample_weight,seeds, params,graph, root):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    metric = params['_metric']
    feature_names = params['feature_names']
    max_degree=params['max_degree']
    sam_pro=params['sam_pro']
    max_samples = int(n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []
    for i in range(n_programs):
        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
            root_arity= None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()
            if method <= sam_pro:
                list_x = [z for z in range(n_features)]
                list_x = set(list_x)
                while True:
                    program, root_arity = parent.sample(random_state, graph, root)
                    if list_x.issubset(set(program)):
                        break
                genome = {'parent_idx': parent_index}
            else:
                # reproduction
                program, root_arity = parent.reproduce()
                genome = {'parent_idx': parent_index}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program,
                           max_degree=max_degree,
                           root_arity=root_arity)

        program.parents = genome
        if genome is not None:
            program.father = parents[genome['parent_idx']].father

        program=program.const_optimize(X,y,sample_weight)

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0

        program.raw_fitness_, graph, root = program.raw_fitness(X, y, curr_sample_weight, graph, root)

        programs.append(program)

    return programs, graph, root


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 init_depth=5,
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div','sin','cos','log','sqrt'),
                 metric='rmse',
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 verbose=0,
                 sam_pro=0.5,
                 graph=[],
                 root=[],
                 n_features=2,
                 max_degree=4,
                 extreme_thre=85,
                 ss_thre=0.9,
                 random_state=None):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.metric = metric
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.verbose = verbose
        self.sam_pro = sam_pro
        self.random_state = random_state
        self.graph = graph
        self.root = root
        self.n_features = n_features
        self.max_degree = max_degree
        self.extreme_thre = extreme_thre
        self.ss_thre = ss_thre
        graph = []

        # initialization of the graph
        row1=[]
        col=[]
        for z in range(len(function_set)+n_features):
            col.append([1e-4,0.0,0.0,0.0,0.0,0.0])
        row1.append(col)
        graph.append(row1)

        for row in range(self.init_depth-2):
            row2 = []
            for i in range(len(function_set)+n_features):
                col=[]
                for z in range(len(function_set)+n_features):
                    col.append([1e-4,0.0,0.0,0.0,0.0,0.0])
                row2.append(col)
            graph.append(row2)

        row3=[]
        for j in range(len(function_set)+n_features):
            col =[]
            for z in range(n_features+1):
                col.append([1e-4,0.0,0.0,0.0,0.0,0.0])
            row3.append(col)
        graph.append(row3)
        self.graph = graph

        # initialization of the root
        root = [[0, 1e-4, 0, 0, 0] for _ in range(self.max_degree)]
        self.root = root

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 )
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)


            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     remaining_time))

    def fit(self, X, y,sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        X, y = self._validate_data(X, y, y_numeric=True)

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman','r2'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if (not isinstance(self.init_depth, int) ):
            raise ValueError('init_depth should be a integer.')

        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_in_,
                                    len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        params = self.get_params()
        params['_metric'] = self._metric
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['graph'] = self.graph
        params['max_degree'] = self.max_degree
        params['sam_pro']= self.sam_pro

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        root_extreme = 0
        for gen in range(prior_generations, self.generations):
            start_time = time()
            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            n_jobs, n_programs, starts = _partition_estimators(self.population_size, 1)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population, self.graph, self.root = _parallel_evolve(n_programs[0],
                                                                   parents, X, y, sample_weight,
                                                                   seeds[starts[0]:starts[1]], params,
                                                                   self.graph, self.root)

            for row in range(len(self.graph)):
                for col in range(len(self.graph[row])):
                    # select extreme
                    extreme = self.graph[row][col][0][-2]
                    if extreme == 0:
                        extreme = calculate_extreme(self.graph[row][col], self.extreme_thre)
                        self.graph[row][col][0][-2] = extreme
                    for i in range(len(self.graph[row][col])):
                        evt = []
                        for j in range(len(self.graph[row][col][i]) - 6):
                            if self.graph[row][col][i][j] > extreme:
                                evt.append(self.graph[row][col][i][j])
                        # calculate shape and scale
                        last_data = self.graph[row][col][i]
                        now_length = len(evt)
                        last_length, last_shape, last_scale = last_data[-5], last_data[-4], last_data[-3]
                        if now_length <= 1:
                            if last_length == 0:
                                self.graph[row][col][i] = self.graph[row][col][i][-6:]
                                self.graph[row][col][i][-6] = 0.01
                            else:
                                self.graph[row][col][i] = self.graph[row][col][i][-6:]
                                self.graph[row][col][i][-6] = "calculate"
                        elif now_length > 1:
                            now_shape, sum = shape(evt, now_length)
                            scale = (1 - now_shape) * sum / (now_length)
                            now_shape = (last_shape * last_length + now_shape * now_length) / (last_length + now_length)
                            scale = (last_scale * last_length + scale * now_length) / (last_length + now_length)
                            self.graph[row][col][i] = []
                            self.graph[row][col][i].extend(["calculate", last_length + now_length, now_shape, scale, extreme, self.ss_thre])
                    # calculate probability
                    for i in range(len(self.graph[row][col])):
                        if self.graph[row][col][i][-6] != 0.01:
                            pa = 1 + self.graph[row][col][i][-4] * self.ss_thre / self.graph[row][col][i][-3]
                            if pa > 0:
                                self.graph[row][col][i][-6] = pa ** (-1 / self.graph[row][col][i][-4])
                            else:
                                self.graph[row][col][i][-6] = random.uniform(0, 0.02)
                    # Normalized probability
                    sum = np.sum(self.graph[row][col][i][-6] for i in range(len(self.graph[row][col])))
                    for i in range(len(self.graph[row][col])):
                        if sum == 0:
                            self.graph[row][col][i][-6] = 1 / len(self.graph[row][col])
                        else:
                            self.graph[row][col][i][-6] = self.graph[row][col][i][-6] / sum

            # select root extreme
            if root_extreme == 0:
                sum = []
                for f in range(len(self.root)):
                    for i in range(len(self.root[f]) - 5):
                        sum.append(self.root[f][i])
                root_extreme = np.percentile(sum, self.extreme_thre)
            for k in range(len(self.root)):
                last_shape, last_length, last_scale = self.root[k][-2], self.root[k][-3], self.root[k][-1]
                evg = []
                for i in range(len(self.root[k]) - 5):
                    if i > root_extreme:
                        evg.append(self.root[k][i])
                now_length = len(evg)
                if now_length <= 1:
                    if last_length == 0:
                        self.root[k] = self.root[k][-5:]
                        self.root[k][-4] = 0.01
                    else:
                        self.root[k] = self.root[k][-5:]
                        self.root[k][-4] = "calculate"
                else:
                    now_shape, sum = shape(evg, now_length)
                    scale = (1 - now_shape) * sum / now_length
                    now_shape = (last_shape * last_length + now_shape * now_length) / (last_length + now_length)
                    scale = (last_scale * last_length + scale * now_length) / (last_length + now_length)
                    self.root[k] = []
                    self.root[k].extend([self.ss_thre, "calculate", last_length + now_length, now_shape, scale])
            # calculate probability
            for i in range(len(self.root)):
                if self.root[i][-4] != 0.01:
                    pa = 1 + self.root[i][-2] * self.ss_thre / self.root[i][-1]
                    if pa > 0:
                        self.root[i][-4] = pa ** (-1 / self.root[i][-2])
                    else:
                        self.root[i][-4] = random.uniform(0, 0.02)
            # Normalized probability
            sum = np.sum(self.root[i][-4] for i in range(len(self.root)))
            for k in range(len(self.root)):
                if sum == 0:
                    self.root[k][-4] = 1 / len(self.root)
                else:
                    self.root[k][-4] = self.root[k][-4] / sum

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            for program in population:
                program.fitness_ = program.raw_fitness_

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

        if self._metric.greater_is_better:
            self._program = self._programs[-1][np.argmax(fitness)]
        else:
            self._program = self._programs[-1][np.argmin(fitness)]

        return self


class SymbolicRegressor(BaseSymbolic, RegressorMixin):


    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 init_depth=5,
                 init_method='half and half',
                 function_set=('root','add', 'sub', 'mul', 'div','sin','cos','log','sqrt','sig'),
                 metric='rmse',
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 verbose=0,
                 sam_pro=0.7,
                 graph=[],
                 root=[],
                 n_features=2,
                 max_degree=4,
                 extreme_thre=85,
                 ss_thre=0.9,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            verbose=verbose,
            graph=graph,
            root=root,
            sam_pro=sam_pro,
            n_features=n_features,
            max_degree=max_degree,
            extreme_thre=extreme_thre,
            ss_thre=ss_thre,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)

        return y


