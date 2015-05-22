#! /usr/bin/env python3

from Bio import PDB
import argparse
import threading


def filter_ligands(pdb_file):
    pdb_parser = PDB.PDBParser(QUIET = True)
    ref = pdb_parser.get_structure(pdb_file, pdb_file)
    for i in ref.get_chains():
        ch = i
    return ch

def is_connected():
    return false

def align_structs(ref, sample):
    ref_atoms = []
    smpl_atoms = []

    for ref_res in ref:
        ref_atoms.append(ref_res['CA'])
    for smpl_res in sample:
        smpl_atoms.append(smpl_res['CA'])

    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, smpl_atoms)
    super_imposer.apply(sample.get_atoms())
    
    return(super_imposer.rms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='folder containing the crystal\
        structure and the predictions', type=str)
    parser.add_argument('ref', help='name of the file containing the crystal\
        structure', type=str)
    args = parser.parse_args()
    
    # test
    ref = filter_ligands(args.ref)
    print(align_structs(ref, ref))

