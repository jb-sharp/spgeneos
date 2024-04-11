"""
@author: Giovanni Bocchi
@email: giovanni.bocchi1@unimi.it
@institution: University of Milan

Utility functions for data saving/loading, for optimization and for plotting.
"""

import os
import random
from itertools import combinations
from functools import partial

import seaborn as sns
import networkx as ntx
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from src.spgeneos import SPNetwork

#%% Data functions

def compute_data(oper, nlist, dlist, savedir, number = 10):
    """
    Precompute operator outputs.

    Parameters:
    ----------
    - oper (list): List of SP GENEOs.
    - nlist (list): Lists of vertices numbers.
    - dlist (list): Lists of degrees.
    - savedir (string): Directory where to save results.
    - number (int, optional): Number of graphs to analyze for each number 
    of vertices. Default is 10.

    Returns:
    -------
    None.

    """
    ngraphs = 0
    npairs = 0
    nglist = []
    nplist = []

    for degree in dlist:
        for vertices in nlist:

            # Generate first K d-regular graphs with n nodes
            graphs_dv = [ntx.random_regular_graph(degree, vertices) for _ in range(1000)]
            graphs_v = [graphs_dv[i] for i in range(number)]

            comb2 = list(combinations(range(len(graphs_v)), 2))

            # Current number of graphs
            ngraphsv = len(graphs_v)
            ngraphs += ngraphsv
            # Current number of pairs
            npairsv = ngraphsv*(ngraphsv - 1) // 2
            npairs += npairsv

            print(f"Computing {npairsv:6d} pairs from ({ngraphsv:4d}) " +
                  f"{degree}-regular graphs with {vertices} vertices")

            matrices = []
            for k, geneo in enumerate(oper):

                # ID for Lambda
                g6id = ntx.to_graph6_bytes(geneo.subgraph,
                                        header = False).decode('UTF-8').replace("\n","")
                model_l = SPNetwork()
                model_l.add([geneo])

                y_file = f"newdata_r{degree}n{vertices}_y_{number}.npz"
                x_file = f"newdata_r{degree}n{vertices}_{g6id}_x_{number}.npz"

                # Compute and save data relative to a single Lambda
                if not os.path.isfile(os.path.join(savedir, y_file)):
                    if ngraphsv >= 2:
                        matrix, _ = model_l.precompute(data = graphs_v,
                                                       progress = tqdm,
                                                       leave = True,
                                                       desc = f"Lambda {k+1}")
                    else:
                        matrix = np.array([])

                    matrices.append(matrix.reshape(-1))
                    np.savez(os.path.join(savedir, x_file),
                             M = matrix)

            if not os.path.isfile(os.path.join(savedir, y_file)):
                matrix_tot = np.stack(matrices).T
                truth = np.ones(matrix_tot.shape[0])
                idx = np.where(matrix_tot.sum(axis = 1) == 0)[0]

                # Checks if the 0s are actually isomorphic graphs
                for i in idx:
                    first, second = comb2[i]
                    exact = int(not ntx.is_isomorphic(graphs_v[first], graphs_v[second]))
                    truth[i] = exact

                np.savez(os.path.join(savedir, y_file),
                         y = truth)

        nglist.append(ngraphs)
        nplist.append(npairs)

    ngl = np.diff([0] + nglist)
    npl = np.diff([0] + nplist)

    result = pd.DataFrame({"Degree d": dlist + ["3,4,5"],
                           "Graph number": list(ngl) + [ngl.sum()],
                           "Pair number": list(npl) + [npl.sum()]}) 

    print()
    print(result)

def load_data(oper, nlist, dlist, loaddir, number = 10):
    """
    Loads precomputed results.

    Parameters:
    ----------
    - oper (list): List of SP GENEOs.
    - nlist (list): Lists of vertices numbers.
    - dlist (list): Lists of degrees.
    - loaddir (string): Directory to load results from.
    - number (int, optional): Number of graphs to analyze for each number 
    of vertices. Default is 10.

    Returns:
    -------
    - matrix (np.array): Matrix of SP GENEO's scores.
    - truth (np.array): Array of ground truths.
    """

    ndata = 0
    for vert in nlist:
        for deg in dlist:
            y_ld = np.load(os.path.join(loaddir, f"newdata_r{deg}n{vert}_y_{number}.npz"))
            ndata += y_ld["y"].shape[0]

    matrix = np.zeros((ndata, len(oper)))
    truth = np.zeros((ndata, 1))

    for i, geneo in enumerate(oper):
        # Get the ID of the current lambda
        g6id = ntx.to_graph6_bytes(geneo.subgraph,
                                header = False).decode('UTF-8').replace("\n","")
        # Stacks arrays for each lambda, for each degree and for each number of vertices
        start, end = 0, 0
        for deg in dlist:
            for vert in nlist:
                x_file = f"newdata_r{deg}n{vert}_{g6id}_x_{number}.npz"
                y_file = f"newdata_r{deg}n{vert}_y_{number}.npz"
                x_ld = np.load(os.path.join(loaddir, x_file))
                y_ld = np.load(os.path.join(loaddir, y_file))

                end += y_ld["y"].shape[0]
                if x_ld["M"].shape[0] > 0:
                    matrix[start : end, i : i + 1] = x_ld["M"]
                    if i == 0:
                        truth[start : end] = y_ld["y"].reshape((len(y_ld["y"]), 1))
                start += y_ld["y"].shape[0]

    # Check that the number of ids matches the number of lambdas
    # assert len(set(lambdas)) == len(nlist)*len(gps)*len(dlist)
    return matrix, truth

#%% Optimization functions

def normalize(array):
    """
    Columwise max normalization of input array.

    Parameters:
    ----------
    - array (np.array): The array to normalize.

    Returns:
    -------
    - result (np.array): The normalized array.

    """
    result = np.zeros(array.shape)
    for j in range(array.shape[1]):
        if array[:, j].max() > 0:
            result[:, j] = array[:, j] / array[:, j].max()
    return result

def log0(array):
    """
    Logarithm clamped to -100 for a zero input.

    Parameters:
    ----------
    - array (np.array): The input array.

    Returns:
    -------
    - result (np.array): The clamped logarithm.

    """
    result = np.zeros(array.shape)
    result[array > 0] = np.log(array[array > 0])
    result[array == 0] = -100*np.ones(array[array == 0].shape)
    return result

def act(array):
    """
    Hyperbolic tangent activation function.

    Parameters:
    ----------
    - array (np.array): The input array.

    Returns:
    -------
    - np.array: The activations.

    """
    return np.tanh(100*array)

def gini(array):
    """
    Compute the Gini index of an array.

    Parameters:
    ----------
    - array (np.array): The input array.

    Returns:
    -------
    - float: The Gini index.

    """
    sorted_list = sorted(array)
    height, area = 0.0, 0.0
    for value in sorted_list:
        height += value
        area += height - value / 2.0
    fair_area = height * len(array) / 2.0
    return (fair_area - area) / fair_area

def sparsity(array):
    """
    Measure of sparsity for a given array.

    Parameters:
    ----------
    - array (np.array): The input array.

    Returns:
    -------
    - float: Array sparsity.

    """
    norm1 = np.abs(array).sum()
    norm2 = np.sqrt((array**2).sum())
    return (np.sqrt(len(array)) - norm1/norm2)/(np.sqrt(len(array)) - 1)

def entropy(alpha, matrix, truth, dim):
    """
    Computes categorical cross-entropy of score matrix @ alpha against the
    gorund truth.

    Parameters:
    ----------
    - alpha (np.array): The convex combination weight. 
    - matrix (np.array): The SP GENEO's results.
    - truth (np.array): The ground truth.
    - dim (int): The number of SP GENEOs considered.

    Returns:
    -------
    - np.array: The resulting loss.

    """
    score = act(matrix @ alpha.reshape((dim, 1)))
    return (-(truth * log0(score) + (1 - truth)*log0(1 - score))).mean()

def optimize_alpha(matrix, truth, toll = 1e-10, selected = None, seed = 1015):
    """
    Solves the constrained optimization problem to find the value of alpha
    minimizing the categorical cross-entropy.
    
        min_{alpha} CrossEntropy(alpha)
            s.t.
                    sum_{i}^p(alpha) = 1
                    alpha[i] >= 0 for every i=1,...,p
                    alpha[i] <= 1 for every i=1,...,p

    Parameters:
    ----------
    - matrix (np.array) The precomputed GENEO results.
    - truth (np.array) The ground truth.
    - toll (float, optional) Tolerance for a nuber to be considered zero. Default is 1e-10.
    - selected (list, optional). Wether to select a subset of SP GENEOs. Default is None.
    - seed (int, optional). Seed for initial guess generation. Default is 1015.

    Returns:
    -------
    - alpha: An optimal weight.

    """

    random.seed(seed)
    np.random.seed(seed)

    if selected:
        matrix = matrix[:, selected]

    ndata, dim = matrix.shape
    matrix = normalize(matrix)
    fun = partial(entropy, matrix = matrix, truth = truth, dim = dim)

    def constraint(alpha):
        return (alpha.reshape((1, dim)) @ np.ones((dim, 1))).item() - 1.0

    constraints = ({'type': 'eq', 'fun': constraint})

    res = minimize(fun,
                   x0 = np.random.rand(dim),
                   method = 'SLSQP',
                   bounds = ((0, None),) * dim,
                   constraints = constraints)

    if res.success:
        alpha = res.x
        confm = confusion_matrix(truth.reshape(-1), np.abs(matrix @ alpha) >= toll)

        print(f"Data size: {matrix.shape[0]:4d}")
        print(f"Optimal loss: {fun(alpha):.3f}\n")
        print("Optimal alpha:")
        for j, alpha_j in enumerate(alpha):
            if j % 6 == 0 and j > 1:
                print()
            print(f'{alpha_j:.3e}', end = '  ')
        print("\n")
        print(f"Squared constraint violation: \n{constraint(alpha)**2:.3g}\n")
        print(f"Non zero weight: {np.where(alpha > toll)[0] + 1}")
        print(f"Number of non zero weight: {len(np.where(alpha > toll)[0])}")
        print(f"Weights concentration: {gini(alpha):.3f}")
        print(f"Weights sparsity: {sparsity(alpha):.3f}\n")
        print(f"Confusion matrix: \n{confm}\n")
        print(f"Accuracy: {np.diag(confm).sum() / ndata:.6f}\n")

    return alpha

def select_forward(oper, nlist, degrees, loaddir, toll = 1e-10):
    """
    Perform a forward selection among the SP GENEOs.

    Parameters:
    ----------
    - oper (list): List of initial SP GENEOs.
    - nlist (list): List of vertices numbers.
    - degrees (list): List of degrees combinations.
    - loaddir (string): Directory to load results from.
    - toll (float, optional) Tolerance for a nuber to be considered zero. Default is 1e-10.

    Returns:
    -------
    selected (list): List of selected operators for each degree combination.

    """
    selected = []

    for i, dlist_i in enumerate(degrees):
        print("Forward Selection:")
        print(f"For k = {dlist_i}")
        matrix_i, truth_i = load_data(oper, nlist, dlist_i, loaddir)
        ilist = []
        accumax = []
        for j in range(len(oper)):
            acculist = []
            for i in range(matrix_i.shape[1]):
                alpha_j = np.ones((j + 1, 1))/(j + 1)
                pred_i = (matrix_i[:, ilist + [i]] @ alpha_j >= toll)
                acculist.append((pred_i == truth_i).mean())
            lp1 = np.argmax(acculist)
            if lp1 not in ilist:
                if j == 0 or (np.max(acculist) > max(accumax)):
                    accumax.append(np.max(acculist))
                    ilist.append(lp1)
                    print(f"Step {j+1}: Selected operators {ilist} " +
                          f"accuracy {np.max(acculist):.6f}")
                else:
                    break
            else:
                break
        selected.append(ilist)
        print()
    return selected

#%% Graphics functions

def accuracy_plot(oper, nlist, dlist, loaddir, axis):
    """
    Plot a bar chart of accuracies for single SP GENEOs.

    Parameters
    ----------
    - oper (list): List of SP GENEOs.
    - nlist (list): List of vertices numbers.
    - dlist (list): List of degrees.
    - loaddir (string): Directory to load results from.
    - axis (plt.axis): Axis to plot on.

    Returns:
    -------
    - accs (np.array): Array of accuracies.

    """
    ticks = range(1, len(oper) + 1)
    accs = np.zeros(len(oper))

    matrix_i, truth_i = load_data(oper, nlist, dlist, loaddir)

    for j in ticks:
        pred = (np.abs(matrix_i[:, j - 1]) > 1e-8).reshape(-1).astype(int)
        accs[j - 1] = np.mean(pred == truth_i.reshape(-1).astype(int))

    result = pd.DataFrame({"Lambda": ticks, "Accuracy": accs})
    sns.barplot(data = result,
                x = "Lambda",
                y = "Accuracy",
                color = "C0")
    axis.set_xlabel(r"$\Lambda_j$")
    axis.set_xticks([0] + list(range(4, len(oper), 5)),
                  [1] + list(range(5, len(oper) + 1, 5)))
    return accs

def accuracy_comparison_plot(oper, nlist, degrees, loaddir, axis):
    """
    Compares the accuracies of single SP GENEOs with repsect to different 
    degree combinations.

    Parameters:
    ----------
    - oper (list): List of SP GENEOs.
    - nlist (list): List of vertices numbers.
    - degreea (list): List of degrees combinations.
    - loaddir (string): Directory to load results from.
    - axis (plt.axis): Axis to plot on.

    Returns:
    -------
    - accs (np.array): Array of accuracies.

    """

    ticks = range(len(oper))
    accs = np.zeros((len(degrees), len(oper)))

    for i, dlist_i in enumerate(degrees):
        matrix_i, truth_i = load_data(oper, nlist, dlist_i, loaddir)
        for j in ticks:
            pred = (np.abs(matrix_i[:, j]) > 1e-8).reshape(-1).astype(int)
            accs[i, j] = np.mean(pred == truth_i.reshape(-1).astype(int))

    for j in range(accs.shape[1]):
        axis.plot([j, j], [0, 1],
                "gray",
                alpha = 0.2)
    for i in range(accs.shape[0]):
        axis.scatter(range(len(oper)), accs[i, :],
                   marker = "D",
                   label = degrees[i])
    axis.set_xticks(ticks, [fr"$\Lambda_{{{i+1}}}$" for i in ticks],
                  rotation = -60)
    plt.legend()
    plt.show()
    return accs

def accuracy_grid_plot(oper, weight, cmap, norm = False, black = False):
    """
    Grid plot of SP GENEO's subgraphs. Can be black or coloured with respect 
    to the accuracy of each SP GENEO.

    Parameters:
    ----------
    - oper (list): List of SP GENEOs.
    - weight (np.array): Weight used to color the subgraphs.
    - cmap (matplotlib.cm): Colormap for plotting accuracies. 
    - norm (bool, optional): Wether to normalize the weight. Default is False.
    - black (cool, optional): Wether to plot in black. Default is False.

    Returns:
    -------
    - fig (matplotlib.figure): The grid plot.

    """

    rows = len(oper) // 10
    if len(oper) % 10 != 0:
        rows = rows + 1

    fig, axis = plt.subplots(rows, 10, figsize = (24, int(2 * (rows + 1))))

    for i in range(10 * rows):
        row = i // 10
        col = i % 10

        try:
            if norm:
                weight_i = weight[i] / weight.max()
            else:
                weight_i = weight[i]

            if not black:
                options_i = {
                "node_size": 60,
                "edgecolors":"black",
                "node_color": to_hex(cmap(weight_i**2)),
                "edge_color": to_hex(cmap(weight_i**2))
                }
            else:
                options_i = {
                "node_size": 60,
                "edgecolors":"black",
                "node_color":"black",
                "edge_color":"black",
                }

            ntx.draw(oper[i].subgraph,
                     ax = axis[row, col],
                     **options_i)

            axis[row, col].set_axis_on()

            if not black:
                axis[row, col].set_xlabel(fr"$A(\Lambda_{{{i+1}}}) = {weight[i].round(3)}$",
                                     fontsize = 15,
                                     labelpad = 2)
            else:
                axis[row, col].set_xlabel(fr"$\Lambda_{{{i+1}}}$",
                                     fontsize = 15,
                                     labelpad = 2)
        except IndexError:
            axis[row, col].set_axis_off()

    return fig
