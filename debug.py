import energies
from Bio.PDB import PDBParser
import os


classes = ["N", "CA", "C", "O", "GCA", "CB", "KNZ", "KCD",
           "DOD", "RNH", "NND", "RNE",
           "SOG", "HNE", "YCZ", "FCZ", "LCD", "CSG"]

parser = PDBParser(QUIET=True)
for caspset in os.listdir("CASP11"):
    try:
        for file in os.listdir(os.path.join("CASP11", caspset)):
            path = os.path.join("CASP11", caspset, file)
            pdb = parser.get_structure("", path)
            for a in pdb.get_atoms():
                if a.name not in ["H", "OXT"] and \
                                energies.atomtype(a) not in classes:
                    print("{},{},{},{}".format(file, a.parent.id[1],
                                               a.parent.resname, a.name))
    except:
        pass