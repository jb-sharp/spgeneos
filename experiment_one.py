"""
@author: Giovanni Bocchi
@email: giovanni.bocchi1@unimi.it
@institution: University of Milan

First experiment.
"""

import random
import numpy as np
import networkx as ntx
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm

from spgeneos import SPGENEO, CycleGENEO, PathGENEO
from subgraphs import triangles, rigids, augmented1, augmented2, augmented3
from utils import compute_data, load_data
from utils import accuracy_plot, accuracy_grid_plot, accuracy_comparison_plot
from utils import optimize_alpha, select_forward

sns.set_theme(context = "paper", style = "white")

#%% Global options

CMAP      = cm.get_cmap("Spectral_r")
NORM      = False
NMAX      = 100
SEED      = 1215
DIR       = "/home/giovanni/Documenti/Dottorato/Articoli e Conferenze/Grafi/Datasets/kreg/cno"

#%% Load subgraphs and subgraphs GENEOs

f_cycles = [CycleGENEO(NMAX, k, norm = NORM) for k in range(3, 11)]
f_stars = [SPGENEO(NMAX, k + 1, ntx.star_graph(k), norm = NORM) for k in range(3, 7)]
f_completes = [SPGENEO(NMAX, k, ntx.complete_graph(k), norm = NORM) for k in range(4, 10)]
f_paths = [PathGENEO(NMAX, k, norm = NORM) for k in range(2, 9)]
f_rigids = [SPGENEO(NMAX, k, subraph, norm = NORM) for subraph, k in rigids]
f_triangles = [SPGENEO(NMAX, k, subraph, norm = NORM) for subraph, k in triangles]
f_augmented1 = [SPGENEO(NMAX, k, subraph, norm = NORM) for subraph, k in augmented1]
f_augmented2 = [SPGENEO(NMAX, k, subraph, norm = NORM) for subraph, k in augmented2]
f_augmented3 = [SPGENEO(NMAX, k, subraph, norm = NORM) for subraph, k in augmented3]

f_augmented = f_augmented1 + f_augmented2 + f_augmented3

operators = f_cycles + f_stars + f_completes + f_paths + f_rigids + f_triangles + f_augmented

#%% Set seed for reproducilibity

random.seed(SEED)
np.random.seed(SEED)

#%% Compute data

nlist = list(range(8, 102, 2))
dlist = [3, 4, 5]
compute_data(operators, nlist, dlist, DIR)

#%% Plot the subgraphs

plot = accuracy_grid_plot(operators,
                          np.zeros(len(operators)),
                          CMAP,
                          black = True)
plot.set_figwidth(16)
plot.set_figheight(11)
plt.tight_layout()

#%% Load data

nlist = list(range(8, 102, 2))
dlist = [3, 4, 5]
degrees = [[3], [4], [5], [3, 4, 5]]
data, truth = load_data(operators, nlist, dlist, DIR)

#%% plot bar char of accuracies

fig, axis = plt.subplots(figsize = (12, 5))
accs = accuracy_plot(operators, nlist, dlist, DIR, axis)
plt.show()

#%% Grid plot of subgraphs and accuracies

plot = accuracy_grid_plot(operators, accs, CMAP)
plot.set_figwidth(16)
plot.set_figheight(11)
plt.tight_layout()

#%% Compute an optimal convex combination of GENEOs
# Plot of the coefficients

fig, axis = plt.subplots(figsize = (12, 5))
accs_all = accuracy_comparison_plot(operators, nlist, degrees, DIR, axis)
alpha = optimize_alpha(data, truth)
accuracy_grid_plot(operators, alpha, CMAP, norm = True)
plot.set_figwidth(16)
plot.set_figheight(11)
plt.tight_layout()
plt.show()

#%% Forward selection
# Plot of the coefficients

selected = sorted(select_forward(operators, nlist, degrees, DIR)[-1])
beta = optimize_alpha(data, truth, selected = selected)

betae = []
k = 0
for i in range(len(operators)):
    if i in selected:
        betae.append(1/len(selected))
        k = k + 1
    else:
        betae.append(0.0)

betae = np.array(betae)
plot = accuracy_grid_plot(operators, betae, CMAP, norm = True)
plot.set_figwidth(16)
plot.set_figheight(11)
plt.tight_layout()
plt.show()
