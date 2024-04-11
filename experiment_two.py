"""
@author: Giovanni Bocchi
@email: giovanni.bocchi1@unimi.it
@institution: University of Milan

Second experiment.
"""

import os
import random
import time
import glob
import numpy as np
import pandas as pd
import networkx as ntx
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

import timeout
from spgeneos import SPGENEO, PathGENEO, CycleGENEO
from subgraphs import triangles, rigids, augmented1, augmented2, augmented3
from kwl import wl_1

#%% Global options

DEGREE    = 3
NNODES    = 100
NSAMPLE   = 100
SEED      = 60
NORM      = False
NMAX      = NNODES
DIR       = "</your/save/directory>"
SELECTED  = [21, 20, 3]
TIMEOUT   = 10


assert DEGREE in list(range(3, 13))

timeout_is_isomorphic = getattr(timeout, f"timeout_is_isomorphic_{TIMEOUT}")
timeout_wl2 = getattr(timeout, f"timeout_wl2_{TIMEOUT}")
timeout_wl3 = getattr(timeout, f"timeout_wl3_{TIMEOUT}")

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

sample = [(ntx.random_regular_graph(DEGREE, NNODES),
           ntx.random_regular_graph(DEGREE, NNODES))
          for _ in range(NSAMPLE)]

i = 0
for G1, G2 in tqdm(sample):

    if not os.path.isfile(DIR + f"/time_acc_{DEGREE}_{NNODES}_{SEED}_{i}.csv"):

        # GENEOs
        tg = []
        yg = []
        for j in range(3):
            sgi = time.time()
            ygi = int(0.5*np.abs(operators[SELECTED[j]](G1) - operators[SELECTED[j]](G2)) >= 1e-10)
            tgi = time.time() - sgi
            yg.append(ygi)
            tg.append(tgi)

        tg1 = tg[0]
        yg1 = yg[0]
        tg2 = sum(tg[0:2])
        yg2 = max(yg[0:2])
        tg3 = sum(tg[0:3])
        yg3 = max(yg[0:3])

        # WL
        sw1 = time.time()
        rw1 = wl_1(G1, G2)
        yw1 = int(not rw1)
        tw1 = time.time() - sw1

        if len(G1.nodes) < 1000:
            sw2 = time.time()
            rw2 = timeout_wl2(G1, G2)
            if rw2 is not None:
                yw2 = int(not rw2)
            else:
                yw2 = rw2
            tw2 = time.time() - sw2
        else:
            tw2 = TIMEOUT
            yw2 = None

        if len(G1.nodes) < 500:
            sw3 = time.time()
            rw3 = timeout_wl3(G1, G2)
            if rw3 is not None:
                yw3 = int(not rw3)
            else:
                yw3 = rw3
            tw3 = time.time() - sw3
        else:
            tw3 = TIMEOUT
            yw3 = None

        # NETWORKX
        sn1 = time.time()
        yn1 = int(not ntx.faster_could_be_isomorphic(G1, G2))
        tn1 = time.time() - sn1

        sn2 = time.time()
        yn2 = int(not ntx.fast_could_be_isomorphic(G1, G2))
        tn2 = time.time() - sn2

        sn3 = time.time()
        yn3 = int(not ntx.could_be_isomorphic(G1, G2))
        tn3 = time.time() - sn3

        sn4 = time.time()
        rn4 = timeout_is_isomorphic(G1, G2)
        if rn4 is not None:
            yn4 = int(not rn4)
        else:
            yn4 = rn4
        tn4 = time.time() - sn4

        
        times = [tg1, tg2, tg3, tn1, tn2, tn3, tn4, tw1, tw2, tw3]
        dec   = [yg1, yg2, yg3, yn1, yn2, yn3, yn4, yw1, yw2, yw3]
        met   = ["GENEO-1", "GENEO-2", "GENEO-3", "NTX-FASTER", "NTX-FAST",
                 "NTX-COULD", "NTX-IS", "1-WL", "2-WL", "3-WL"]
        deg   = [DEGREE]*len(times)
        siz   = [NNODES]*len(times)
        tru   = [max(d for d in dec if d is not None)]*len(times)

        if max(tru) > 0:
            result = pd.DataFrame({"Time": times,
                                   "Decision": dec,
                                   "Truth": tru,
                                   "Method": met,
                                   "Degree": deg,
                                   "Size": siz})

            result.to_csv(DIR + f"/time_acc_{DEGREE}_{NNODES}_{SEED}_{i}.csv")
            i = i + 1
            del result, times, dec
        else:
            del times, dec
    else:
        i = i + 1

#%% Evaluate results

DIR = "</your/save/directory>"
sns.set_theme(context = "talk", style = "white")
PNAME = "husl"
NCOLORS = 6
PALETTE = sns.color_palette(PNAME, NCOLORS)
CMAP = [PALETTE[0]]*3 + [PALETTE[1]]*3 + [PALETTE[2]]*1 + [PALETTE[4]]*3

csvs = sorted(glob.glob(os.path.join(DIR, "time_acc_*_*_*_*.csv")))
dfs = [pd.read_csv(f) for f in csvs]
data = pd.concat(dfs, ignore_index = True)
data["Accuracy"] = data["Decision"] == data["Truth"]
data["Count"] = np.ones(len(data), dtype = int)

grouped = data.groupby(by = ["Degree", "Size", "Method"])[["Count","Time","Accuracy",]].sum()
grouped["Time"] = grouped["Time"] / grouped["Count"]
grouped["Accuracy"] = grouped["Accuracy"] / grouped["Count"]

#%% Plot mean execution times

times = sns.relplot(data = data,
                    x = "Size",
                    y = "Time",
                    hue = "Method",
                    style = "Method",
                    col = "Degree",
                    kind = "line",
                    palette = CMAP,
                    markersize = 12,
                    markers = True,
                    dashes = False,
                    legend = True)

# Linear reference
(times.map(sns.lineplot, x = [100, 500, 1000, 5000, 10000], 
           y = [100*1e-6, 500*1e-6, 1000*1e-6, 5000*1e-6, 10000*1e-6], 
           color="black", 
           dashes=(1, 2), 
           zorder=0))

# Quadratic reference
(times.map(sns.lineplot, x = [100, 500, 1000, 5000, 10000], 
           y = [100**2*1e-6, 500**2*1e-6, 1000**2*1e-6, 5000**2*1e-6, 10000**2*1e-6], 
           color="black", 
           dashes=(6, 2), 
           zorder=0))

times.set_titles(r"$d$ = {col_name}")
times.set_ylabels("Time (s)")
times.set_xlabels(r"Size $N$")
plt.yscale('log')
plt.xscale('log')
plt.show()

#%% Plot mean accuracies

NCMAP = [CMAP[i] for i in range(len(CMAP)) if i != 6]
accu = sns.relplot(data = data[data["Method"] != "NTX-IS"],
                   x = "Size",
                   y = "Accuracy",
                   hue = "Method",
                   style = "Method",
                   col = "Degree",
                   kind = "line",
                   palette = NCMAP,
                   markersize = 12,
                   markers = True,
                   dashes = False,
                   legend = True)

accu.set_titles(r"$d$ = {col_name}")
accu.set_ylabels("Accuracy")
accu.set_xlabels(r"Size $N$")
plt.yscale('log')
plt.xscale('log')
plt.show()
