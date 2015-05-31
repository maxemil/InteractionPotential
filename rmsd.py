#! /usr/bin/env python3

from Bio import PDB
import argparse
import pandas as pd

from energies import *


def alignstructures(crystal, pdbpath):
    """
    Align predicted Sructure to crystal structure and compute
    the RMSE
    :param crystal: pdb object
    :param pdb: pdb object
    """

    parser = PDB.PDBParser(QUIET=True)
    pdb = parser.get_structure("", pdbpath)
    ligandfilter(pdb)

    crystal_atoms = []
    pdb_atoms = []

    for crystal_res in crystal.get_residues():
        crystal_atoms.append(crystal_res['CA'])
    for pdb_res in pdb.get_residues():
        pdb_atoms.append(pdb_res['CA'])

    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(crystal_atoms, pdb_atoms)
    super_imposer.apply(pdb.get_atoms())
    print(super_imposer.rms)
    return super_imposer.rms

