#!/bin/usr/env python3


import matplotlib.pyplot as plt
import pandas as pd


def histogram(data, title):
    plt.clf()
    plt.hist(data, bins=25)
    plt.title("Distribution of Energies")
    plt.xlabel("kcal/mol")
    plt.suptitle("Dataset: {}".format(title))
    plt.savefig("{}.pdf".format(title))


sets = ["T0762", "T0769", "T0776", "T0784"]

for x in sets:
    frame = pd.read_csv("{}.csv".format(x))
    histogram(frame[frame.columns[1]], x)
