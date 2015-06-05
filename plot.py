#!/bin/usr/env python3


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def histogram(data, title, addition, unit="kcal/mol"):
    plt.clf()
    plt.hist(data, bins=25)
    plt.title("Distribution of {}".format(addition))
    plt.xlabel(unit)
    plt.suptitle("Dataset: {}".format(title))
    plt.savefig("results/{}_{}.pdf".format(addition, title))


def rmsds_cmp(subset, energies, rmsds):
    #Convert names in rmsds
    def shorten(string):
        return string.split("/")[-1]
    rmsds["PDB"] = rmsds["PDB"].apply(shorten)
    # Unify
    matrix = pd.DataFrame.join(energies.set_index("PDB"),
                               rmsds.set_index("PDB"), how="inner")
    # Print results
    print("Subset {}".format(subset))

    # Value correlation
    print("Pearson Corelation (values): {}".format(pearsonr(
        matrix["RMSD"], matrix["Energy in kcal/mol"])[0]))
    print("Spearman Corelation (rank): {}".format(spearmanr(
        matrix["RMSD"], matrix["Energy in kcal/mol"])[0]))

    print("Top 5 according to RMSD")
    print(matrix.sort("RMSD").iloc[0:5])
    print("Top 5 according to Interaction Potential")
    print(matrix.sort("Energy in kcal/mol").iloc[0:5])



sets = ["T0762", "T0769", "T0776", "T0784"]

for subset in sets:
    energies = pd.read_csv("results/{}.csv".format(subset))
    rmsds = pd.read_csv("results/rmsds_{}".format(subset), sep="\t")

    histogram(energies["Energy in kcal/mol"], subset, "Energies")
    histogram(rmsds["RMSD"], subset, "RMSDs", unit="Å")

    rmsds_cmp(subset, energies, rmsds)
