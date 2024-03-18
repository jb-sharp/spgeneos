"""
@author: Giovanni Bocchi
@email: giovanni.bocchi1@unimi.it
@institution: University of Milan

Functions to stop the execution of a function after a certain time.
"""

import signal
import networkx as ntx
from kwl import wl_k

class TimeoutException(Exception):
    """
    Class for a TimeoutException
    """

def break_after(seconds):
    """
    Function decorator to stop execution after a certain amount of seconds.

    Parameters
    ----------
    seconds (int): The timeout in seconds.

    Raises
    ------
    TimeoutException

    Returns
    -------
    function (function): The decorated function.

    """

    def timeout_handler(signum, frame):
        raise TimeoutException

    def function(function):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                res = function(*args, **kwargs)
                signal.alarm(0)
                return res
            except TimeoutException:
                return None
            return None
        return wrapper
    return function

@break_after(10)
def timeout_is_isomorphic_10(graph1, graph2):
    """
    Apply a timeout of 10 seconds to nx.is_isomorphic

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return ntx.is_isomorphic(graph1, graph2)
@break_after(10)
def timeout_wl2_10(graph1, graph2):
    """
    Apply a timeout of 10 seconds to 2-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 2)
@break_after(10)
def timeout_wl3_10(graph1, graph2):
    """
    Apply a timeout of 10 seconds to 3-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 3)

@break_after(15)
def timeout_is_isomorphic_15(graph1, graph2):
    """
    Apply a timeout of 15 seconds to nx.is_isomorphic

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return ntx.is_isomorphic(graph1, graph2)
@break_after(15)
def timeout_wl2_15(graph1, graph2):
    """
    Apply a timeout of 15 seconds to 2-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 2)
@break_after(15)
def timeout_wl3_15(graph1, graph2):
    """
    Apply a timeout of 15 seconds to 3-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 3)

@break_after(30)
def timeout_is_isomorphic_30(graph1, graph2):
    """
    Apply a timeout of 30 seconds to nx.is_isomorphic

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return ntx.is_isomorphic(graph1, graph2)
@break_after(30)
def timeout_wl2_30(graph1, graph2):
    """
    Apply a timeout of 30 seconds to 2-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 2)
@break_after(30)
def timeout_wl3_30(graph1, graph2):
    """
    Apply a timeout of 30 seconds to 3-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 3)

@break_after(45)
def timeout_is_isomorphic_45(graph1, graph2):
    """
    Apply a timeout of 45 seconds to nx.is_isomorphic

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return ntx.is_isomorphic(graph1, graph2)
@break_after(45)
def timeout_wl2_45(graph1, graph2):
    """
    Apply a timeout of 45 seconds to 2-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 2)
@break_after(45)
def timeout_wl3_45(graph1, graph2):
    """
    Apply a timeout of 45 seconds to 3-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 3)

@break_after(60)
def timeout_is_isomorphic_60(graph1, graph2):
    """
    Apply a timeout of 60 seconds to nx.is_isomorphic

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return ntx.is_isomorphic(graph1, graph2)
@break_after(60)
def timeout_wl2_60(graph1, graph2):
    """
    Apply a timeout of 60 seconds to 2-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 2)
@break_after(60)
def timeout_wl3_60(graph1, graph2):
    """
    Apply a timeout of 60 seconds to 2-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 3)

@break_after(90)
def timeout_is_isomorphic_90(graph1, graph2):
    """
    Apply a timeout of 60 seconds to nx.is_isomorphic

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return ntx.is_isomorphic(graph1, graph2)
@break_after(90)
def timeout_wl2_90(graph1, graph2):
    """
    Apply a timeout of 60 seconds to 2-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 2)
@break_after(90)
def timeout_wl3_90(graph1, graph2):
    """
    Apply a timeout of 60 seconds to 3-WL

    Parameters:
    ----------
    - graph1 (nx.Graph): The first graph.
    - graph2 (nx.Graph): The second graph.

    Returns
    -------
    - bool: Result of the test.

    """
    return wl_k(graph1, graph2, k = 3)
