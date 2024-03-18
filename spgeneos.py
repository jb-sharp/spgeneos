"""
@author: Giovanni Bocchi
@email: giovanni.bocchi1@unimi.it
@institution: University of Milan

Defines abstract classes for SP GENEOs looking for subgraphs. 
Path and Cycles have special implemetations based on a modified
breadth first search.
"""

from math import factorial
from itertools import combinations, permutations

import numpy as np
import networkx as ntx
from scipy.special import binom
from sympy.combinatorics import Permutation
from networkx.algorithms.isomorphism.isomorphvf2 import GraphMatcher

class SPGENEO():
    """
    A class that models a GENEO based on a generic Subgraph Permutant.
    
    Attributes:
    -----------
        nnodes (int): The number of nodes of an input graph.
        knodes (int): The number of nodes in the subgraph.
        subgraph (nx.Graph): The subgraph.
        norm (bool): Whether to normalize the result of the opertor.
    
    Methods:
    -------
        __init__(nnodes, knodes, subgraph, norm=True)
            Initialize the class with the given parameters. 
        isogroup()
            Computes the cardinality of the group of self isomorphisms of the subgraph.
        mean0(array)
            Compute the mean of a list of numbers, but ignore any NaN values.
        sum0(array)
            Compute the sum of a list of numbers, but ignore any NaN values.
        __call__(graph)
            Computes the operator on the graph graph.
    """

    def __init__(self, nnodes, knodes, subgraph, norm = True):
        """
        Initialize the SPGENEO class with the specified parameters.
        
        Parameters:
        -----------
        - nnodes (int): The number of nodes of an input graph.
        - knodes (int, optional): The number of nodes in the subgraph.
        - subgraph (nx.Graph, optional): The subgraph.
        - norm (bool, optional): Whether to normalize the output score. Default is True.
        """

        super().__init__()

        assert isinstance(subgraph, ntx.classes.graph.Graph)
        assert len(subgraph.nodes) == knodes

        self.norm = norm
        self.nnodes = nnodes
        self.subgraph = subgraph
        self.knodes = knodes
        self.const = int(binom(self.nnodes, len(self.subgraph))*factorial(len(self.subgraph)))
        self.nedges = len(subgraph.edges)

        if self.norm:
            self.const1 = 1.0/self.const
        else:
            self.const1 = 1.0

    def isogroup(self):
        """
        Computes the cardinality of the group of self isomorphisms of the subgraph.
                
        Returns:
        --------
        - group (list): The group.
        - count (int): The cardinality of the group.
        """
        lnodes = len(self.subgraph.nodes)
        symmetric = [Permutation(p) for p in permutations(range(lnodes), lnodes)]

        count = 0
        group = []

        for permutation in symmetric:
            for edge in self.subgraph.edges:
                adj = (permutation(edge[0]), permutation(edge[1])) in self.subgraph.edges
                if not adj:
                    break
            if adj:
                group.append(permutation)
                count = count + 1

        return group, count

    def mean0(self, array):
        """
        Compute the mean of an array a. If the array is empty, return 0.
        
        Parameters:
        -----------
        - array (array-like): The input array.
        
        Returns:
        --------
        - output (float): The mean of the array.
        """
        array = np.array(array)
        if array.size == 0:
            output = 0.0
        else:
            output = np.nanmean(array)

        return output

    def sum0(self, array):
        """
        Compute the sum of an array a. If the array is empty, return 0.
        
        Parameters:
        -----------
        - array (array-like): The input array.
        
        Returns:
        --------
        - output (float): The sum of the array.
        """
        array = np.array(array)
        if array.size == 0:
            output = 0.0
        else:
            output = np.sum(array)

        return output

    def __call__(self, graph):
        """
        Compute the score of a graph G against this SPGENEO class instance.
        
        Parameters:
        -----------
        - graph (nx.Graph): A NetworkX graph to be scored.
        
        Returns:
        --------
        - output (float): The score of the graph.
        """

        maxg = max(d for i, d in graph.degree)
        maxl = max(d for i, d in self.subgraph.degree)
        if len(graph.nodes) >= len(self.subgraph.nodes) and (maxl <= maxg):
            mappings = GraphMatcher(graph, self.subgraph).subgraph_isomorphisms_iter()
            leng = sum(1 for m in mappings)
            output = leng*self.const1
        else:
            output = 0.0

        return output

class PathGENEO():
    """
    A class that models a GENEO based on a path Subgraph Permutant.
   
    Attributes:
    -----------
        nnodes (int): The number of nodes of an input graph.
        knodes (int): The number of nodes in the path subgraph.
        norm (bool): Whether to normalize the result of the opertor.
   
    Methods:
    --------
        __init__(nnodes, knodes, norm=True)
            Initialize the class with the given parameters.
        traverse(graph, visit, depth, level=0)
            Given a graph, return all possible paths from one node 
            to an other of length depth that do not contain any repeated nodes.
        __call__(graph)
            Computes the operator on the input graph.    
    """

    def __init__(self, nnodes, knodes, norm = True):
        """
       Initialize the PathGENEO class with the specified parameters.
       
       Parameters:
       ----------
       - nnodes (int): The number of nodes of an input graph.
       - knodes (int, optional): The number of nodes in the path subgraph.
       - norm (bool, optional): Whether to normalize the output score. Default is True.
       """

        assert knodes >= 2

        super().__init__()
        self.norm = norm
        self.nnodes = nnodes
        self.subgraph = ntx.path_graph(knodes)
        self.knodes = knodes
        self.const = int(binom(self.nnodes, len(self.subgraph))*factorial(len(self.subgraph)))
        self.nedges = len(self.subgraph.edges)

        if self.norm:
            self.const1 = 1.0/self.const
        else:
            self.const1 = 1.0

    def traverse(self, graph, visit, depth, level = 0):
        """
        Given a graph, return all possible paths from one node to an other of 
        length depth that do not contain any repeated nodes.
        
        Parameters:
        -----------
        - graph (nx.Graph): A NetworkX graph.
        - visit (list): The current path.
        - depth (int): The desired path length.
        - level (int, optional): The current depth of the recursion. Default is 0.
        
        Yields:
        -------
        - visit (list): A list of nodes representing a possible path.
        """
        if level < depth:
            current = visit[-1]
            actions = list(graph.neighbors(current))
            for i in visit[:-1]:
                actions = [action for action in actions if action not in graph.neighbors(i)]
            for action in actions:
                yield from self.traverse(graph, visit + [action], depth, level + 1)
        else:
            yield visit

    def __call__(self, graph):
        """
        Compute the score of a graph against this PathGENEO class instance.
        
        Parameters:
        -----------
        - G (nx.Graph): A NetworkX graph to be scored.
        
        Returns:
        --------
        - output (float): The score of the graph.
        """

        maxg = max(d for i, d in graph.degree)
        maxl = max(d for i, d in self.subgraph.degree)
        if len(graph.nodes) >= len(self.subgraph.nodes) and (maxl <= maxg):
            count = 0
            for i in graph.nodes:
                count += len(list(self.traverse(graph, [i], self.knodes)))
            output = count*self.const1
        else:
            output = 0.0

        return output

class CycleGENEO():
    """
    A class that models a GENEO based on a cycle Subgraph Permutant.
   
    Attributes:
    -----------
        nnodes (int): The number of nodes of an input graph.
        knodes (int): The number of nodes in the cycle subgraph.
        norm (bool): Whether to normalize the result of the opertor.
   
    Methods:
    --------
        __init__(nnodes, knodes, norm=True)
            Initialize the class with the given parameters.
        loop(graph, visit, depth, level=0)
            Given a graph, return all possible paths from a node to an other of
            length depth that do not contain any repeated nodes but the first 
            and the last that must coincide.
        __call__(graph)
            Computes the operator on the input graph.    
    """

    def __init__(self, nnodes, knodes, norm = True):
        """
        Initialize the CycleGENEO class with the specified parameters.
        
        Parameters:
        -----------
        - nnodes (int): The number of nodes of an input graph.
        - knodes (int, optional): The number of nodes in the cycle subgraph.
        - norm (bool, optional): Whether to normalize the output score.
        Default is True.
        """

        assert knodes >= 3

        super().__init__()
        self.norm = norm
        self.nnodes = nnodes
        self.subgraph = ntx.cycle_graph(knodes)
        self.knodes = knodes
        self.const = int(binom(self.nnodes, len(self.subgraph))*factorial(len(self.subgraph)))
        self.nedges = len(self.subgraph.edges)

        if self.norm:
            self.const1 = 1.0/self.const
        else:
            self.const1 = 1.0

    def loop(self, graph, visit, depth, level = 0):
        """
        Given a graph, return all possible paths from a node to an other of
        length depth that do not contain any repeated nodes but the first 
        and the last that must coincide.
        
        Parameters:
        -----------
        - graph (nx.Graph): A NetworkX graph.
        - visit (list): The current cycle.
        - depth (int): The desired cycle length.
        - level (int, optional): The current depth of the recursion. 
        Default is 0.
        
        Yields:
        -------
        - visit (list): A list of nodes representing a possible cycle.
        """
        if level < depth - 2:
            current = visit[-1]
            actions = list(graph.neighbors(current))
            for i in visit[0:-1]:
                actions = [action for action in actions
                           if action not in graph.neighbors(i)
                           and action not in visit]
            for action in actions:
                yield from self.loop(graph, visit + [action], depth, level + 1)
        elif level == depth - 2:
            current = visit[-1]
            actions = list(graph.neighbors(current))
            for i in visit[1:-1]:
                actions = [action for action in actions
                           if action not in graph.neighbors(i)
                           and action not in visit]
            for action in actions:
                yield from self.loop(graph, visit + [action], depth, level + 1)
        elif level == depth - 1:
            current = visit[-1]
            actions = [action for action in graph.neighbors(current)
                       if action == visit[0]]
            for action in actions:
                yield from self.loop(graph, visit + [action], depth, level + 1)
        else:
            if visit[0] == visit[-1]:
                yield visit

    def __call__(self, graph):
        """
        Compute the score of a graph against this CycleGENEO class instance.
        
        Parameters:
        -----------
        - G (nx.Graph): A NetworkX graph to be scored.
        
        Returns:
        --------
        - output (float): The score of the graph.
        """

        maxg = max(d for i, d in graph.degree)
        maxl = max(d for i, d in self.subgraph.degree)
        if len(graph.nodes) >= len(self.subgraph.nodes) and (maxl <= maxg):
            unique = set()
            for i in graph.nodes:
                for cycle in self.loop(graph, [i], self.knodes):
                    unique.add(tuple(cycle))
            output = len(unique)*self.const1
        else:
            output = 0.0

        return output

class SPNetwork():
    """
    A class that models a network os SP GENEOS.
    
    Methods:
    --------
        add(perms)
            Add a list of SP GENEOs to the network.
        count()
            Return the number of SP GENEOs in the network.
        __call__(x)
            Given a graph x, return the network result.
        precompute(data=None, leave=False, progress=iter, desc="")
            Precompute network's results on given data. 
    """

    def __init__(self):
        pass

    def add(self, perms):
        """
        Add the given SP GENEOs to the network.
        
        Parameters:
        -----------
        - perms (list): A list of SP GENEOs to be added.
        """
        count = self.count()
        for i, perm in enumerate(perms):
            setattr(self, f"perm{count + i}", perm)

    def count(self):
        """
        Count the number of SP GENEOs in the network.
       
        Returns:
        --------
        - count (int): The number of SP GENEOs.
        """
        count = 0
        while hasattr(self, f"perm{count}"):
            count = count + 1
        return count

    def __call__(self, graph):
        """
        Compute the score of a graph G against this GenNetwork class instance.
        
        Parameters:
        -----------
        - graph (nx.Graph): A NetworkX graph to be scored.
        
        Returns:
        - output (float): The network score of the graph.
        """
        output = []
        for i in range(self.count()):
            output.append(getattr(self, f"perm{i}")(graph))
        return output

    def precompute(self, data, leave = False, progress = iter, desc = ""):
        """
        Compute the scores for the given dataset.
        
        Parameters:
        -----------
        - data (list): A list of NetworkX graphs to be scored.
        - leave (bool, optional): Whether to display progress bars when computing
        scores. Default is True.
        - progress (iterable, optional): An iterable object for displaying 
        progress bars. Default is iter.
        - desc (str, optional): The description of the progress bar. 
        Default is an empty string.
        
        Returns:
        - tuple: A tuple containing the scores and ground truth labels.
        """

        noper = self.count()
        ndata = len(data)

        matrix1 = np.zeros((ndata, noper))

        i = 0
        for graph in progress(data, desc = desc, leave = leave):
            for j in range(noper):
                matrix1[i, j] = getattr(self, f"perm{j}")(graph)
            i = i + 1

        comb2 = list(combinations(range(ndata), 2))
        matrix = np.zeros((len(comb2), noper))

        i = 0
        for (first, second) in iter(comb2):
            for j in range(noper):
                matrix[i, j] = 0.5 * np.abs(matrix1[first, j] - matrix1[second, j])
            i = i + 1
        return matrix, np.ones((len(comb2), 1))
    