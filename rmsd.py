#!/usr/bin/env python3


from Bio import PDB
import pandas as pd
from energies import ligandfilter,updateprogress,waiting

import argparse
import os
from multiprocessing import Pool


def backbonec(crystal, pdb):
    """
    Iterator over all CA atoms in a PDB structure.
    PDB consists of only one model and only one chain.
    :param crystal: PDB.Structure.Structure
    :param pdb: PDB.Structure.Structure
    :return: iterator
    """
    # Both pdbs have only one chain
    crystalchain = next(crystal.get_chains())

    for pdbres in pdb.get_residues():
        try:
            crystalres = crystalchain.child_dict[pdbres.id]
            yield [crystalres.child_dict['CA'], pdbres.child_dict['CA']]
        except KeyError:
            # Residue seems to be missing in crystal
            continue


def prepare_workload(crystal, workload):
    for (nr, pdbpath) in enumerate(workload):
        yield [crystal.copy(), pdbpath, nr / len(workload)]


def alignstructures(crystal, pdbpath, stat):
    """
    Align predicted Sructure to crystal structure and compute
    the RMSE
    :param crystal: PDB.Structure.Structure
    :param pdbpath: str
    :param stat: float
    :return: float
    """
    updateprogress(pdbpath, stat)

    parser = PDB.PDBParser(QUIET=True)
    pdb = parser.get_structure("", pdbpath)
    ligandfilter(pdb)
    # Match CAs
    c_atoms = pd.DataFrame(list(backbonec(crystal, pdb)),
                           columns=['crystal', 'pdb'])
    # Structure Align
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

    async = pool.starmap_async(alignstructures,
                               prepare_workload(crystal, workload))
    waiting(async)
    rmsd = async.get()

    pool.close()
    # Make 100% appear
    updateprogress("Finished", 1)
    # Store output
    with open(out, "w") as handler:
        handler.write("PDB\tRMSD\n")
        for pdb, value in zip(workload, rmsd):
           handler.write("{}\t{}\n".format(pdb, value))
           #handler.write(str(workload[i]) + '\t' +  str(rmsd[i]) + '\n')


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
