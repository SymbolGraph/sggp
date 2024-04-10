

from copy import copy
import random

import numpy as np


from sklearn.utils.random import sample_without_replacement
from .functions import _Function
from .utils import check_random_state
from scipy.optimize import minimize

class _Program(object):

    """A program-like representation of the evolved program.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    n_features : int
        The number of features in `X`.

    metric : _Fitness object
        The raw fitness metric.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 metric,
                 random_state,
                 root_arity,
                 max_degree,
                 feature_names=None,
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = init_depth
        self.init_method = init_method
        self.n_features = n_features
        self.metric = metric
        self.feature_names = feature_names
        self.program = program
        self.root_arity = root_arity
        self.max_degree = max_degree

        if self.program is None:
            # Create a naive random program
            list_x = [i for i in range(n_features)]
            list_x = set(list_x)
            while True:
                self.program = self.build_program(random_state,self.max_degree)
                if list_x.issubset(set(self.program)):
                    break

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
        self.father = None

    def const_optimize(self, X, y, sample_weight=None):
        self.program[0].arity = self.root_arity
        const_sum = 0
        list = []
        for i in range(len(self.program)):
            if isinstance(self.program[i], float):
                const_sum += 1
                list.append(i)
        if const_sum == 0:
            return self
        x0 = [1.0 for _ in range(const_sum)]

        def f(x0):
            node = self.program[0]
            if isinstance(node, float):
                return np.repeat(node, X.shape[0])
            if isinstance(node, int):
                return X[:, node]
            apply_stack = []
            i = -1
            father = []
            for node in self.program:
                i += 1
                if isinstance(node, _Function):
                    apply_stack.append([i, node])
                else:
                    # Lazily evaluate later
                    apply_stack[-1].append(node)
                    father.append([i, apply_stack[-1][0]])
                while len(apply_stack[-1]) - 1 == apply_stack[-1][1].arity + 1:
                    # Apply functions that have sufficient arguments
                    function = apply_stack[-1][1]
                    pos = 0
                    terminals = []
                    for t in apply_stack[-1][2:]:
                        if isinstance(t, float):
                            terminals.append(np.repeat(x0[pos], X.shape[0]))
                            pos += 1
                        elif isinstance(t, int):
                            terminals.append(X[:, t])
                        else:
                            terminals.append(t)
                    intermediate_result = function(*terminals)
                    if len(apply_stack) != 1:
                        list = []
                        apply_stack.pop()
                        for j in apply_stack[::-1]:
                            list.append(j[0])
                        apply_stack[-1].append(intermediate_result)
                    else:
                        result = self.metric(y,intermediate_result, sample_weight)
                        return result
        result = minimize(f, x0)
        optimized_constants = result.x

        for i in range(len(list)):
            self.program[list[i]] = optimized_constants[i]
        return self

    def build_program(self, random_state,max_degree):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        max_depth = self.init_depth
        # Start a program with a function to avoid degenerative programs
        function = self.function_set[0]
        function.arity = random_state.randint(1, max_degree)
        self.root_arity = function.arity
        program = [function]
        terminal_stack = [function.arity]
        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (choice + 1 <= len(self.function_set)):
                function = random_state.randint(1, len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                terminal = random_state.randint(self.n_features + 1)
                if terminal == self.n_features:
                    terminal = 1.0
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""

        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []
        mediate = []
        i = -1
        father = []
        self.program[0].arity = self.root_arity

        for node in self.program:
            i += 1
            if isinstance(node, _Function):
                apply_stack.append([i, node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)
                father.append([i, apply_stack[-1][0]])

            while len(apply_stack[-1]) - 1 == apply_stack[-1][1].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][1]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                else t for t in apply_stack[-1][2:]]
                intermediate_result = function(*terminals)
                mediate.append([apply_stack[-1][0], intermediate_result])

                if len(apply_stack) != 1:
                    start = apply_stack[-1][0]
                    list = []
                    apply_stack.pop()
                    for j in apply_stack[::-1]:
                        list.append(j[0])
                    father.append([start] + list)
                    apply_stack[-1].append(intermediate_result)
                else:
                    for i in range(len(father)):
                        if father[i][-1] != 0:
                            for j in father:
                                if j[0] == father[i][-1]:
                                    father[i] = father[i] + j[1:]
                                    break
                    self.father = father
                    return intermediate_result, self.father
        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None, random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program), self.root_arity

    def raw_fitness(self, X, y, sample_weight, graph, degree):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """

        y_pred, father = self.execute(X)
        raw_fitness = self.metric(y, y_pred, sample_weight)
        tree_ss = self.semantic_similarity(y_pred, y)

        arity = self.root_arity
        degree[arity - 1].insert(len(degree[arity - 1]) - 5, tree_ss)

        for tuple in father:
            layer = len(tuple) - 2
            start = tuple[0]
            end = tuple[1]
            function = self.program[start]

            if layer<len(graph)-1:
                if isinstance(function, _Function):
                    start = self.function_set.index(function)
                elif isinstance(function, int):
                    start = function +len(self.function_set)
                else:
                    start = self.n_features +len(self.function_set)
            #last layer
            else:
                if isinstance(function, _Function):
                    start = self.function_set.index(function)
                elif isinstance(function, int):
                    start = function + 1
                else:
                    start = self.n_features + 1
            terminal = self.program[end]
            end = self.function_set.index(terminal)
            if layer == 0:
                graph[layer][layer][start - 1].insert(len(graph[layer][layer][start - 1]) - 6, tree_ss)
            else:
                graph[layer][end - 1][start - 1].insert(len(graph[layer][end - 1][start - 1]) - 6, tree_ss)

        return raw_fitness, graph, degree
    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
            # Choice of crossover points follows Koza's (1992) widely used approach
            # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())
        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end
    def greedy(self, random_state, graph, start, layer):
        ram = random_state.uniform(0, 1)
        p = 0
        max_index = 0
        flag = True
        for i in range(len(graph[layer][start])):
            p = p + graph[layer][start][i][-6]
            if ram <= p:
                max_index = i
                flag = False
                break
        if flag == True:
            max_index = random_state.randint(0, len(graph[layer][start]))
        return max_index

    def rootsample(self, random_state, graph, root):
        max_depth = self.init_depth
        # Start a program with a function to avoid degenerative programs
        function = self.function_set[0]
        ram = random_state.uniform(0, 1)
        p = 0
        flag = True
        for i in range(len(root)):
            p = p + root[i][-4]
            if ram <= p:
                function.arity = i + 1
                flag = False
                break
        if flag == True:
            function.arity = random_state.randint(0, len(root)) + 1
        program = [function]
        terminal_stack = []
        terminal_stack.append([0, function.arity])
        root_arity = function.arity
        while terminal_stack:
            depth = len(terminal_stack)
            # Determine if we are adding a function or terminal
            function = self.greedy(random_state, graph, terminal_stack[-1][0], depth - 1)
            if depth < max_depth and function < len(self.function_set) - 1:
                function = self.function_set[function + 1]
                program.append(function)
                terminal_stack.append([self.function_set.index(function) - 1, function.arity])
            else:
                terminal = function
                if depth == max_depth:
                    if terminal == self.n_features:
                        terminal = 1.0
                else:
                    if terminal == self.n_features + len(self.function_set) - 1:
                        terminal = 1.0
                    else:
                        terminal = terminal - len(self.function_set) + 1
                program.append(terminal)
                terminal_stack[-1][1] -= 1
                while terminal_stack[-1][1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program, root_arity
                    terminal_stack[-1][1] -= 1

        # We should never get here
        return None, None

    def sample(self, random_state, graph, root):
        # Get a subtree to replace
        self.program[0].arity = self.root_arity
        start, end = self.get_subtree(random_state)
        if start == 0:
            program, root_arity = self.rootsample(random_state, graph, root)
            return program, root_arity

        for i in self.father:
            if i[0] == start:
                height = len(i) - 1
                root_v = i[1]
                break
        root_v = self.program[root_v]
        if isinstance(root_v, _Function):
            root_index = self.function_set.index(root_v) - 1
        elif isinstance(root_v, int):
            root_index = root_v
        else:
            root_index = self.n_features
        # Get a subtree to donate
        donor = [root_v]
        terminal_stack = [[root_index, 1]]
        while terminal_stack:
            depth = len(terminal_stack)
            # Determine if we are adding a function or terminal
            max_index = self.greedy(random_state, graph, terminal_stack[-1][0], height - 2 + depth)
            if depth < self.init_depth - height + 1 and max_index < len(self.function_set) - 1:
                function = self.function_set[max_index + 1]
                donor.append(function)
                terminal_stack.append([max_index, function.arity])
            else:
                if depth == self.init_depth - height + 1:
                    if max_index == self.n_features:
                        max_index = 1.0
                else:
                    if max_index == self.n_features + len(self.function_set) - 1:
                        max_index = 1.0
                    else:
                        max_index = max_index - len(self.function_set) + 1
                donor.append(max_index)
                terminal_stack[-1][1] -= 1
                while terminal_stack[-1][1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        break
                    terminal_stack[-1][1] -= 1
        # Insert genetic material from donor
        return (self.program[:start] + donor[1:] + self.program[end:]), self.root_arity


    def calculate_histogram(self, x, y, xbins=10, ybins=10):
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        x_interval = (xmax - xmin) / xbins
        y_interval = (ymax - ymin) / ybins
        x_in_bini = []
        y_in_binj = []
        xhist, yhist = np.zeros(xbins), np.zeros(ybins)
        L = np.zeros((xbins, ybins))
        for i in range(xbins):
            x_margin = (xmin + i * x_interval, xmin + (i + 1) * x_interval)
            if i == xbins - 1:
                x_in_bini.append(x >= x_margin[0])
            else:
                x_in_bini.append(np.logical_and(x >= x_margin[0], x < x_margin[1]))
            xhist[i] = np.sum(x_in_bini[i])
        for j in range(ybins):
            y_margin = (ymin + j * y_interval, ymin + (j + 1) * y_interval)
            if j == ybins - 1:
                y_in_binj.append(y >= y_margin[0])
            else:
                y_in_binj.append(np.logical_and(y >= y_margin[0], y < y_margin[1]))
            yhist[j] = np.sum(y_in_binj[j])
        for i in range(xbins):
            for j in range(ybins):
                xy_in_binij = np.logical_and(x_in_bini[i], y_in_binj[j])
                L[i, j] = np.sum(xy_in_binij)
        return xhist, yhist, L

    def semantic_similarity(self,output, target):
        """
            output: torch.Tensor. Predictions of the SR model with shape=(batch_size, 1)
            target: torch.Tensor. Target value with shape=(batch_size, 1)
        """
        N = len(output)
        x, y = output, target
        xhist, yhist, L = self.calculate_histogram(x, y, xbins=int(np.sqrt(N)), ybins=int(np.sqrt(N)))
        px = xhist / N
        py = yhist / N
        pxy = L / N
        I = np.sum(pxy * np.log(pxy / (np.outer(px, py) + 1e-6) + 1e-6))
        H = -1 * np.sum((py * np.log(py + 1e-6)))
        ss = max(I / H, 0)
        return ss

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
