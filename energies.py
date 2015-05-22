#!/usr/bin/env python3


from Bio import PDB
import argparse
import os
import pandas as pd
from collections import defaultdict
from itertools import islice


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
                        ("HIS", "CE1"), ("TRP", "NE2")],
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
                        #("ILE", "CD"),
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


def ispossiblepair(atom1, atom2):
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
    if respos2 - respos1 <= connectivitymatrix.loc[code1, code2]:
        return False
    return True


def energylookup(atom1, atom2):
    """
    Lookup energy codes for a pair of atoms.
    :param atom1: PDBAtom
    :param atom2: PDBAtom
    :return:float
    """
    return contactenergies.loc[atomtype(atom1), atomtype(atom2)]


def itercontactpairs(pdb):
    """
    Iterator over all possible contact pairs in a PDB structure.
    PDB consists of only one model and only one chain.
    :param pdb: PDB.
    :return:iterator
    """
    heavyatoms = [i for i in pdb.get_atoms() if i.element != "H"]
    for (index, atom1) in enumerate(heavyatoms):
        # Second indices from iterator slice (islice)
        for atom2 in islice(heavyatoms, index + 1, None):
            if ispossiblepair(atom1, atom2):
                yield (atom1, atom2)


def computecontactenergy(pdbpath):
    """
    Compute contact pairs energy for a PDB file.
    :param pdbpath: str
    :return:float
    """
    parser = PDB.PDBParser()
    pdb = parser.get_structure("", pdbpath)
    ligandfilter(pdb)
    # Sum up energies
    sum = 0
    for pair in itercontactpairs(pdb):
        try:
            sum += energylookup(*pair)
        except KeyError:
            print("Couldn't lookup", pair)
    print("Energy is {} kcal/mol at T=298K".format(sum/21))


def main(dir, out):
    pass


if __name__ == "__main__":
    shell = argparse.ArgumentParser()
    shell.add_argument("path",
                       help="Path to directory that contains the PDB files",
                       type=str)
    shell.add_argument("out", help="Output-file for the computed energies")
    args = shell.parse_args()

    main(args.path, args.out)