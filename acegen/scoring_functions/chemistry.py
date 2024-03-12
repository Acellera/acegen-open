from rdkit.Chem import AllChem as Chem
from rdkit.Chem.QED import qed


def QED(smiles: list):
    """Calculate QED score for a list of SMILES strings."""
    rewards = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rewards.append(qed(mol))
        else:
            rewards.append(0.0)
    return rewards
