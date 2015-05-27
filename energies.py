#!/usr/bin/env python3


from Bio import PDB
import pandas as pd

import argparse
import os
import time
import sys
from math import floor
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool


################################################################################
# PARAMETERS
################################################################################

"""
Special cases, defined as list of (amino acid, atom)-tuples.
All greek letters are substituted by these following ascii equivalents:
            -alpha: A
            -beta: B
            -gamma: G
            -delta: D
            -epsilon: E
            -eta: H
            -zeta: Z
See table 1 in paper.
"""
specialcodes = {"GCA": [("GLY", "GA")],
                "CB": [("PRO", "CG"), ("PRO", "CD")],
                "KNZ": [("LYS", "CE"), ("LYS", "NZ")],
                "KCD": [("LYS", "CD")],
                "DOD": [("ASP", "CG"), ("ASP", "OD1"), ("ASP", "OD2"),
                        ("GLU", "CD"), ("GLU", "OE1"), ("GLU", "OE2")],
                "RNH": [("ARG", "CZ"), ("ARG", "NH1"), ("ARG", "NH2")],
                "NND": [("ASN", "CG"), ("ASN", "OD1"), ("ASN", "ND2"),
                        ("GLN", "CD"), ("GLN", "OE1"), ("GLN", "NE2")],
                "RNE": [("ARG", "CD"), ("ARG", "NE")],
                "SOG": [("SER", "CB"), ("SER", "OG"), ("THR", "OG1"),
                        ("TYR", "OH")],
                "HNE": [("HIS", "CG"), ("HIS", "ND1"), ("HIS", "CD2"),
                        ("HIS", "CE1"), ("HIS", "NE2"), ("TRP", "NE1")],
                "YCZ": [("TYR", "CE1"), ("TYR", "CE2"), ("TYR", "CZ")],
                "FCZ": [("ARG", "CG"), ("GLN", "CG"), ("GLU", "CG"),
                        ("ILE", "CG1"), ("LEU", "CG"), ("LYS", "CG"),
                        ("MET", "CG"), ("MET", "SD"), ("PHE", "CG"),
                        ("PHE", "CD1"), ("PHE", "CD2"), ("PHE", "CE1"),
                        ("PHE", "CE2"), ("PHE", "CZ"), ("THR", "CG2"),
                        ("TRP", "CG"), ("TRP", "CD1"), ("TRP", "CD2"),
                        ("TRP", "CE2"), ("TRP", "CE3"), ("TRP", "CZ2"),
                        ("TRP", "CZ3"), ("TRP", "CH2"), ("TYR", "CG"),
                        ("TYR", "CD1"), ("TYR", "CD2")],
                "LCD": [("ILE", "CG2"),
                        # The paper suggests the nomenclature CD for Isoleucin,
                        # but it seems that CD1 is the standard.
                        # See: http://www.bmrb.wisc.edu/ref_info/atom_nom.tbl
                        # ("ILE", "CD"),
                        ("ILE", "CD1"),
                        #
                        ("LEU", "CD1"),
                        ("LEU", "CD2"), ("MET", "CE"), ("VAL", "CG1"),
                        ("VAL", "CG2")],
                "CSG": [("CYS", "SG")]
                }


# Values of the connectivity matrix as defined in the paper.
connectivitymatrix = pd.DataFrame([[3, 3, 2, 2],
                                   [3, 3, 3, 3],
                                   [4, 3, 3, 3],
                                   [3, 3, 2, 2]],
                                  index=[1, 2, 3, 4], columns=[1, 2, 3, 4])


# Class of atoms for connectivity matrix, default is side-chain, class 4.
connectivityclass = defaultdict(lambda: 4,
                                {"N": 1,
                                 "CA": 2,
                                 "C": 3,
                                 "O": 3})


# Load contact energies
contactenergies = pd.DataFrame.from_csv("contactenergies.csv")
# Fill missing values such that the matrix is symmetric
contactenergies = contactenergies.fillna(contactenergies.T)


################################################################################
# //PARAMETERS
################################################################################


# VT100 controls
CURSOR_UP_ONE = "\x1b[1A"
CURSOR_DOWN_ONE = "\x1b[1B"
ERASE_LINE = "\x1b[2K"
DELETE_LINE = "\x1b[1M"


"""
Pre-process special cases for performance boost.
"""
processed_specialcases = dict()
for code, caseslist in specialcodes.items():
    for case in caseslist:
        processed_specialcases[case] = code


def atomtype(atom):
    """
    Atom type for atom. (Table 1)
    :param atom:PDB.Atom
    :return:str
    """
    try:
        # Constant time lookup of special cases
        return processed_specialcases[(atom.parent.resname, atom.name)]
    except KeyError:
        # For backbone atoms, atom.name is code
        return atom.name


def ligandfilter(pdb):
    """
    Remove water and other ligands from pdb.
    :return:None
    """
    # Remove non amino acid residues
    # To upkeep the integrity due to detaching, iterate over child_list copy!
    for model in pdb.child_list[:]:
        for chain in model.child_list[:]:
            for res in chain.child_list[:]:
                if not PDB.is_aa(res):
                    chain.detach_child(res.id)
            if len(chain) == 0:
                model.detach_child(chain)
        if len(model) == 0:
            pdb.detach_child(model)
    # There is only one model left
    assert len(pdb) == 1
    # This model has only one chain
    assert len(pdb.child_list[0]) == 1


def iscontactpair(atom1, atom2):
    """
    Determine whether two atoms are possible contact pairs based on their
    connection class. It is assumed that the atoms are on the same chain.
    :param atom1: PDB.Atom
    :param atom2: PDB.Atom
    :return:bool
    """
    if atom1 - atom2 < 6:
        # Exclusion due to close proximity
        return False
    # Keep if residue position is distant enough
    code1 = connectivityclass[atom1.name]
    code2 = connectivityclass[atom2.name]
    respos1 = atom1.parent.id[1]
    respos2 = atom2.parent.id[1]
    return respos2 - respos1 > connectivitymatrix.loc[code1, code2]


def energylookup(atom1, atom2):
    """
    Lookup energy codes for a pair of atoms.
    :param atom1: PDBAtom
    :param atom12: PDBAtom
    :return:float
    """
    # Attribute atomtype set by itercontactpairs function
    return contactenergies.loc[atomtype(atom1), atomtype(atom2)]


def allpairs(pdb):
    """
    Iterator over all heavy atom pairs in a PDB structure.
    PDB consists of only one model and only one chain.
    :param pdb: PDB.
    :return:iterator
    """
    # Filter out hydrogen and C-Terminal hydroxyl-group
    heavyatoms = []
    for atom in pdb.get_atoms():
        if atom.element != "H" and atom.name != "OXT":
            heavyatoms.append(atom)
    # Generator of pairs
    for atom1, atom2 in combinations(heavyatoms, 2):
        yield (atom1, atom2)


def worker(pair):
    """
    Check whether a pair is a valid contact pair, and compute energies.
    :param pair: tuple(PDBAtom, PDBAtom)
    :return: flaot
    """
    if iscontactpair(*pair):
        return energylookup(*pair)
    else:
        return 0


def computecontactenergy(pdbpath, pool):
    """
    Compute contact pairs energy for a PDB file.
    :param pdbpath: str
    :param pool: multiporecessing.Pool
    :return:float (energy in kcal/mol)
    """
    parser = PDB.PDBParser(QUIET=True)
    pdb = parser.get_structure("", pdbpath)
    ligandfilter(pdb)
    # Compute pairwise energy on multiple cores

    async = pool.map_async(worker, allpairs(pdb))
    # Wating wheel for user
    waiting(async)
    energies = async.get()
    # Convert to kcal/mol
    return sum(energies) / 21


def waiting(async):
    """
    Show waiting symbol as long as async is still running.
    :param async: Async object
    :return:
    """
    print()
    state = 0
    chars = r"|/-\|"
    while not async.ready():
        print(CURSOR_UP_ONE + DELETE_LINE + chars[state])
        state = (state + 1) % 5
        time.sleep(0.5)
    print(DELETE_LINE)


def updateprogress(current, ratio):
    """
    Let user know how much work has been completed so far, and what the current
    file is.
    :param current: str
    :param ratio: flaot
    :return:
    """
    # Delete last progressbar
    print(CURSOR_UP_ONE * 2 + DELETE_LINE * 2, end="")
    # Print new one
    print("[{:3.2f}% {: <50}]".format(ratio * 100, "-" * floor(ratio * 50)))
    print("Current PDB: {}".format(current))


def main(path, out, cores):
    """
    Compute contact energies for each pdb in path and write results to 'out'.
    :param path: str
    :param out: str
    :param cores: int
    :return:
    """
    # Find all pdbs in path
    workload = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1].lower() == ".pdb":
            workload.append(file)
    # Print few newlines to prevent progressbar from messing up the shell
    print("\n\n")
    # Compute energies
    pool = Pool(processes=cores)
    results = []
    for (nr, pdb) in enumerate(workload):
        updateprogress(pdb, nr / len(workload))
        e = computecontactenergy(os.path.join(path, pdb), pool)
        results.append((pdb, e))
    pool.close()
    # Store output
    with open(out, "w") as handler:
        handler.write("PDB,Energy in kcal/mol\n")
        for pair in results:
            print("{},{}\n".format(*pair))


if __name__ == "__main__":
    shell = argparse.ArgumentParser()
    shell.add_argument("path",
                       help="Path to directory that contains the PDB files",
                       type=str)
    shell.add_argument("out", help="Output-file for the computed energies")
    shell.add_argument("cores", help="Nr. of cores", type=int, default=4)
    args = shell.parse_args()
    main(args.path, args.out, args.cores)