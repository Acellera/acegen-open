from rdkit.Chem import AllChem as Chem
from rdkit.Chem import QED

def QED(smiles: list):
    rewards = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rewards.append(QED.qed(mol))
        else:
            rewards.append(0.0)
    return rewards