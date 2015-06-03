#!/usr/bin/env python3


from Bio import PDB
import pandas as pd
from energies import ligandfilter,updateprogress,waiting

import argparse
import os
import time
import sys
from multiprocessing import Pool

def backbonec(crystal, pdb):
    """
    Iterator over all CA atoms in a PDB structure.
    PDB consists of only one model and only one chain.
    :param pdb: PDB.
    :return:iterator
    """
    for pres in pdb.get_residues():
        for cres in crystal.get_residues():
            if cres.id ==pres.id:
                yield [cres.child_dict['CA'], pres.child_dict['CA']]

def enumerate_pdb(crystal, workload):
    for (nr, pdb) in enumerate(workload):
        yield [crystal, pdb, nr/len(workload)]


def alignstructures(crystal, pdbpath, stat):
    """
    Align predicted Sructure to crystal structure and compute
    the RMSE
    :param crystal: pdb object
    :param pdb: pdb object
    """

    parser = PDB.PDBParser(QUIET=True)
    pdb = parser.get_structure("", pdbpath)
    ligandfilter(pdb)
    updateprogress(pdbpath, stat)


    c_atoms = pd.DataFrame(columns=['crystal','pdb'])
    for (nr, (pc, cc)) in enumerate(backbonec(crystal, pdb)):
        c_atoms.loc[nr] = [pc, cc]
        
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(c_atoms['crystal'], c_atoms['pdb'])
    super_imposer.apply(pdb.get_atoms())

    return super_imposer.rms


def main(path, out, cores, crystal):
    """
    Compute contact energies for each pdb in path and write results to 'out'.
    :param path: str
    :param out: str
    :param cores: int
    :return:
    """

    parser = PDB.PDBParser(QUIET=True)
    crystal = parser.get_structure("", crystal)
    ligandfilter(crystal)

    # Find all pdbs in path
    workload = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1].lower() == ".pdb":
            workload.append(os.path.join(path, file))
    # Print few newlines to prevent progressbar from messing up the shell
    print("\n\n")
    # Compute energies
    pool = Pool(processes=cores)

    async = pool.starmap_async(alignstructures, enumerate_pdb(crystal, workload))
    waiting(async)
    rmsd = async.get()

    pool.close()
    # Make 100% to appear
    updateprogress("Finished", 1)
    # Store output
    with open(out, "w") as handler:
        handler.write("PDB\tRMSD\n")
        for i in range(len(rmsd)):
           handler.write(str(workload[i]) + '\t' +  str(rmsd[i]) + '\n')


if __name__ == "__main__":
    shell = argparse.ArgumentParser()
    shell.add_argument("path",
                       help="Path to directory that contains the PDB files",
                       type=str)
    shell.add_argument("out", help="Output-file for the computed energies")
    shell.add_argument("cores", help="Nr. of cores", type=int, default=4)
    shell.add_argument("crystal", help="name of the file containing the true\
                       crystal structure", type=str)
    args = shell.parse_args()
    main(args.path, args.out, args.cores, args.crystal)
