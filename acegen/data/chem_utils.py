import numpy as np

from rdkit.Chem import AllChem as Chem, Draw


def get_mol(smiles_or_mol):
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    elif isinstance(smiles_or_mol, Chem.rdchem.Mol):
        return smiles_or_mol
    else:
        return None


def fraction_valid(mol_list):
    parsed_mols = []
    for mol in mol_list:
        mol = get_mol(mol)
        if mol:
            parsed_mols.append(1)
        else:
            parsed_mols.append(0)
    return np.mean(parsed_mols)


def draw(mol_list, molsPerRow=5, subImgSize=(300, 300)):
    mols = [get_mol(mol) for mol in mol_list]
    image = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=subImgSize)

    return image
