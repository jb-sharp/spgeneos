"""
@author: Giovanni Bocchi
@email: giovanni.bocchi1@unimi.it
@institution: University of Milan

Defines functions that implement the k-WL isomorphism test.
Based on NetworkX implementation of 1-WL.
"""

from collections import Counter
from itertools import product
from hashlib import blake2b

import networkx as ntx

def _hash_label(label, digest_size):
    """
    Hash one label.
    
    Parameters:
    ----------
    - label (string): The label to be hashed.
    - digest_size (int): Size (in bits) of blake2b hash digest.
    
    Returns:
    --------
    - string: Label hash.
    """
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

def _init_tuple_labels(graph, tuples, indices):
    """
    Compute intial labels of tuples.
    
    Parameters:
    ----------
    - graph (nx.Graph): The graph to be considered.
    - tuples (list): List of tuples.
    - indices (int): List of tuples indices.
    
    Returns:
    --------
    - labels (dict): Dictionary of initial tuple labels.
    """

    labels = {}
    for tup in tuples:
        degrees = [str(graph.degree(i)) for i in tup]
        connections = [str(int((tup[a], tup[b]) in graph.edges)) for (a, b) in indices]
        label = "".join(degrees + connections)
        labels[tup] = label

    return labels

def _k_aggregate(graph, tup, tuple_labels, k):
    """
    Compute new labels for given tuple.
    
    Parameters:
    ----------
    - graph (nx.Graph): The graph to be considered.
    - t (tuple):  The current tuple.  
    - tuple_labels (list): Labels before aggregation.
    - k (int): Size of tuples to be considered.
    
    Returns:
    --------
    - string: Tuple label after the aggregation.
    """

    label_list = []
    for i in range(k):
        i_label_list = []
        for node in graph.nodes:
            nbr = tuple(tup[j] if j != i else node for j in range(k))
            i_label_list.append(tuple_labels[nbr])
        label_list.append(i_label_list)

    return tuple_labels[tup] + "".join(["".join(sorted(i_label_list))
                                        for i_label_list in label_list])

def kwl_graph_hash(graph, iterations = 3, digest_size = 16, k = 2):
    """
    Return k-Weisfeiler Lehman (k-WL) graph hash.

    Hashes are identical for isomorphic graphs and strong guarantees that
    non-isomorphic graphs will get different hashes.
    
    Parameters:
    ----------
    - graph (nx.Graph): The graph to be hashed.
    - iterations (int): Number of neighbor aggregations to perform.
    Should be larger for larger graphs, Default is 3.  
    - digest_size (int): Size (in bits) of blake2b hash digest to use 
    for hashing node labels. Default is 16
    - k (int): Size of tuples to be considered, Default is 2

    Returns:
    -------
    h (string): Hexadecimal string corresponding to hash of the input graph.
    """

    def kwl_step(graph, labels, tuples, k):
        """
        Perfom one step of the algorithm. 
        
        First aggregates than hashes the new labels.
        
        Parameters:
        ----------
        - graph (nx.Graph): The graph to be hashed.
        - labels (list): Current labels.  
        - tuples (list): list of tuples.
        - k (int): Size of tuples to be considered.

        Returns:
        -------
        new_labels (list): New labels.
        """
        new_labels = {}
        for tup in tuples:
            label = _k_aggregate(graph, tup, labels, k)
            new_labels[tup] = _hash_label(label, digest_size)
        return new_labels

    # Set tuples and indices
    tuples = list(product(*([graph.nodes]*k)))
    indices = list(product(range(k), range(k)))

    # Set initial node labels
    tuple_labels = _init_tuple_labels(graph, tuples, indices)

    subgraph_hash_counts = []
    for _ in range(iterations):
        tuple_labels = kwl_step(graph, tuple_labels, tuples, k)
        counter = Counter(tuple_labels.values())

        # sort the counter, extend total counts
        subgraph_hash_counts.extend(sorted(counter.items(), key=lambda x: x[0]))

    # hash the final counter
    return _hash_label(str(tuple(subgraph_hash_counts)), digest_size)

def wl_1(graph1, graph2):
    """
    1-WL test as a binary operator.
    
    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns:
    -------
    - bool: Result of the test.

    """
    hash1 = ntx.weisfeiler_lehman_graph_hash(graph1)
    hash2 = ntx.weisfeiler_lehman_graph_hash(graph2)
    return hash1 == hash2

def wl_k(graph1, graph2, k = 2):
    """
    k-WL test as a binary operator.

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.
    - k (int, optional): The order of the test, Default is 2.

    Returns:
    -------
    - bool: Result of the test.

    """
    hash1 = kwl_graph_hash(graph1, k = k)
    hash2 = kwl_graph_hash(graph2, k = k)
    return hash1 == hash2
