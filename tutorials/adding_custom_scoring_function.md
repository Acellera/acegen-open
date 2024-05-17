# Tutorial: Integrating Custom Scoring Functions in AceGen

---

## Defining a custom scoring function

Defining custom scoring functions can be of interest to users who want to use a scoring function that is not included in the MolScore library, our default scoring library. In this tutorial, we will demonstrate how to define a custom scoring function and use it in the AceGen scripts.

As an example, we will define a custom scoring function to evaluate the quality of the generated molecules. We will use the QED score as the custom scoring function. The QED score is a measure of drug-likeness of a molecule. It is a real number between 0 and 1, with 1 being the most drug-like. The QED score is calculated using the RDKit library.


```python 
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
```

The signature of any custom scoring function should be a function that accepts a list of SMILES strings and returns a list of floats, where each float is the score of the corresponding SMILES string.

## Using a custom scoring function in the training scripts (Option 1)

Once our QED scoring function is defined, we can add it to the `custom_scoring_functions` dictionary in the `acegen/scoring_functions/__init__.py` file. The dictionary of scoring functions is used in the training scripts to map the name of the scoring function to the actual function. We have already added the QED scoring function as an example for this tutorial. 

```python
from acegen.scoring_functions import custom_scoring_functions

custom_scoring_functions = {
   "QED": QED,
}

for k, v in custom_scoring_functions.items():
    print(k, ":" , v)
```

The output of the above code is therefore:

```QED : <function QED at 0x7f9a092cf790>```

Finally, we can specify that we want to use the QED scoring function in any the configuration file. 
To do this, we need to set the `molscore` parameter to `null` and the `custom_task` parameter to the name of the scoring function. For example, we can modify the `acegen/scripts/reinvent/config_denovo.yaml` file as follows:

```yaml
...
molscore: null
molscore_include: null
custom_task: QED
...
```

## Using a custom scoring function in the training scripts (Option 2)

It is also possible to use the scpring function directly without modifying the internal code of ACEGEN.
For that, we can provide the path to our factory to the config parameter `custom_task` directly.

```yaml
...
molscore: null
molscore_include: null
custom_task: myproject.my_scoring_funcions.QED
...
```

## Training with our scoring function

Finally, we can run the training script as usual. The training script will use the QED scoring function to evaluate the quality of the generated molecules.

```bash
python acegen/scripts/reinvent/reinvent.py --config-name config_denovo
```
