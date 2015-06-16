#!/usr/bin/env python2.7
"""
Run script within pymol
Compute atomic packaging of internal atoms
"""


from pymol import cmd, stored
import scipy as sp
from itertools import product
import os
import pandas as pd


CORES = 4
#PATH = ".../Assignment 2/supplement02"
PATH = ".../project_3_pdbs/"
# to be evaluated radii
THRESHOLD = 13
RADS = sp.linspace(2, THRESHOLD, 25)

def packing(pdb):
    "Derive mean packing density of pdb as pd.Series."
    cmd.delete('all')
    cmd.load(pdb)
    cmd.remove('solvent')
    # Only heavy atoms
    cmd.remove('hydro')
    # Compute SAS per atom
    cmd.set('dot_solvent', 1)
    cmd.get_area('all', load_b=1)
    cmd.select('interior', 'b = 0')

    counts = pd.Series(0, index=RADS)
    vest = pd.Series(0, index=RADS)
    # from biggest to smallest radius
    for r in RADS[::-1]:
        # Counting
        counts.loc[r] = cmd.select('extended', 'interior extend {}'.format(r))
        cmd.remove('not extended')
        # moleculare area
        #cmd.set('dot_solvent', 0)
        vest[r] = cmd.get_area('all')
    # Results
    cvdens = counts / vest
    counts.index = ["{}_rawcount".format(i) for i in counts.index]
    vest.index = ["{}_volume estimate".format(i) for i in vest.index]
    cvdens.index = ["{}_cv density".format(i) for i in cvdens.index]
    return pd.concat(([counts, cvdens, vest]))


def euclid_step(a, b):
    d = sp.sqrt(sum([(a[i] - b[i]) ** 2 for i in range(3)]))
    if d < THRESHOLD:
        for x in RADS:
            if x >= d:
                return x
    else:
        return None


def slowpacking(pdb):
    "Derive mean packing density of pdb as pd.Series."
    cmd.delete('all')
    cmd.load(pdb)
    cmd.remove('solvent')
    # Only heavy atoms
    cmd.remove('hydro')
    # Compute SAS per atom
    cmd.set('dot_solvent')
    cmd.get_area('all', load_b=1)
    N = float(cmd.select('interior', 'b = 0'))

    internal_coords = [at.coord for at in cmd.get_model('interior').atom]#[1:50]
    all_coords = [at.coord for at in cmd.get_model('all').atom]#[1:50]

    # Count
    counts = pd.Series(0, index=RADS)
    for a, b in product(internal_coords, all_coords):
        es = euclid_step(a, b)
        if es is not None:
            counts.loc[es] += 1
    counts = counts.cumsum()
    # Mean per center atom
    meancounts = counts / N
    # Normalize to density
    volumina = pd.Series(4 / 3.0 * sp.pi * (RADS ** 3), index=RADS)
    density = meancounts / volumina
    # Results
    counts.index = ["{}_correctcount".format(i) for i in counts.index]
    density.index = ["{}_density".format(i) for i in density.index]
    return pd.concat(([counts, density]))


cmd.cd(PATH)
result = []
pdbs = []
for pdb in os.listdir("."):
    if not pdb.endswith(".pdb"):
        continue
    pdbs.append(pdb)
    result.append(slowpacking(pdb))
    #break
result = pd.DataFrame(result, index=pdbs)
result.to_csv("packing.csv")