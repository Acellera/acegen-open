# AceGen: A TorchRL-based toolkit for reinforcement learning in generative chemistry

## Overview

In recent years, reinforcement learning (RL) has been increasingly used in drug design to propose molecules with specific properties under defined constraints. However, RL problems are inherently complex, featuring independent and interchangeable components with diverse method signatures and data requirements.

AceGen applies TorchRL - a modern general decision-making library that provides well-integrated reusable components - to make a robust toolkit tailored for generative drug design.

![Alt Text](./acegen/images/train_zaleplon.png)
![Alt Text](./acegen/images/chem_zaleplon.png)

## Installation

### Conda environment

To create the conda / mamba environment, run

    conda create -n acegen python=3.10 -y
    conda activate acegen
    pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
    
### Install Tensordict

To install Tensordict, run

    git clone https://github.com/pytorch/tensordict.git
    cd tensordict
    python setup.py install

### Install TorchRL

To install TorchRL, run

    git clone https://github.com/pytorch/rl.git
    cd rl
    python setup.py install

### Install MolScore

To install MolScore, run

    pip3 install rdkit func_timeout dask distributed pystow zenodo_client matplotlib scipy pandas joblib seaborn molbloom Levenshtein
    git clone https://github.com/MorganCThomas/MolScore.git
    cd molscore
    git checkout develop
    python setup.py install

### Install AceGen

To install AceGen, run

    pip3 install tqdm wandb hydra-core
    cd acegen-open
    python setup.py install


## Running training scripts

To run the training scripts, run

    python scripts/a2c/a2c.py
    python scripts/ppo/ppo.py
    python scripts/reinvent/reinvent.py
    python scripts/ahc/ahc.py

To modify training parameters, edit the corresponding YAML file in each example's directory.

# Available models

We provide a variety of example priors that can be selected in the configuration file. These include:

- A Gated Recurrent Unit (GRU) model
  - pre-training dataset1 (default): [ChEMBL](https://www.ebi.ac.uk/chembl/)
  - pre-training dataset2: [ZINC250k](https://github.com/wenhao-gao/mol_opt/blob/main/data/zinc.txt.gz)
  - umber of parameters: 4,363,045


- A Long Short-Term Memory (LSTM) model
  - pre-training dataset: [ChEMBL](https://www.ebi.ac.uk/chembl/)
  - number of parameters: 5,807,909
 

- A GPT-2 model
  - pre-training dataset: [REAL 350/3 lead-like, 613.86M cpds, CXSMILES](https://enamine.net/compound-collections/real-compounds/real-database-subsets)
  - number of parameters: 5,030,400

# Changing the scoring function

To change the scoring function, adjust the `molscore` parameter in any configuration files. Set it to point to a valid 
MolScore configuration file (e.g.  ../MolScore/molscore/configs/GuacaMol/Albuterol_similarity.json). 
Alternatively, you can set the `molscore` parameter to the name of a valid MolScore benchmark 
(such as MolOpt, GuacaMol, etc.) to automatically execute each task in the benchmark. For further details on MolScore, 
please refer to the [MolScore](https://github.com/MorganCThomas/MolScore) repository.

Alternatively, training scripts can be edited to use any custom scoring function.
The following example demonstrates how to use a custom scoring function:

    from rdkit.Chem import AllChem, QED
    from acegen.rl_env import SMILESEnv
    from acegen.vocabulary import SMILESVocabulary
    from torchrl.collectors import RandomPolicy
    
    # Create a vocabulary from a list of characters
    chars = ["START", "END", "(", ")", "1", "=", "C", "N", "O"]
    chars_dict = {char: index for index, char in enumerate(chars)}
    vocab = SMILESVocabulary.create_from_dict(chars_dict, start_token="START", end_token="END")
        
    def evaluate_mol(smiles: str):
        mol = AllChem.MolFromSmiles(smiles)
        if mol:
            return QED(mol)
        else:
            return 0.0
    
    # Define a function to evaluate a list of molecules
    # Should accept a list of SMILES strings and return a list or array of floats
    def evaluate_mols(smiles: list):
        return [evaluate_mol(smi) for smi in smiles]
    
    # Generate molecules
    env =  SMILESEnv(
        start_token=vocab.start_token_index,
        end_token=vocab.end_token_index,
        length_vocabulary=len(vocab),
        batch_size=1,
    )
    data = env.rollout(max_steps=100)

    # Use the custom scoring function to compute the rewards
    # These lines can be found in the training scripts, so if the scoring function is changed, it should be updated accordingly
    smiles_str = [vocab.decode(smi.numpy()) for smi in data["action"]]
    reward = evaluate_mols(smiles_str)

# Integration of custom models

We encourage users to integrate their own models into AceGen, modifying the existing code as needed.

`/acegen/models/gru.py` and `/acegen/models/lstm.py` offer methods to create RNNs of varying sizes, which can be use
to load custom models. Similarly, `/acegen/models/gpt2.py` can serve as a template for integrating HuggingFace models. 

# Results on the [MolOpt](https://arxiv.org/pdf/2206.12411.pdf) benchmark

Algorithm comparison for the Area Under the Curve (AUC) of the top 100 molecules on MolOpt benchmark scoring functions. 
Each algorithm ran 5 times with different seeds, and results were averaged. We used the default configuration for each algorithm, including the GRU model for the prior.

| Task                      | Reinvent | AHC   | A2C   | PPO   | PPOD  |
|---------------------------|----------|-------|-------|-------|-------|
| Albuterol_similarity      | 0.569    | 0.640 | 0.760 | **0.873** | 0.862 |
| Amlodipine_MPO            | 0.506    | 0.505 | 0.511 | 0.522 | **0.556** |
| C7H8N2O2                  | 0.615    | 0.563 | 0.737 | 0.858 | **0.870** |
| C9H10N2O2PF2Cl            | 0.556    | 0.553 | 0.610 | 0.679 | **0.704** |
| Celecoxxib_rediscovery    | 0.566    | 0.590 | **0.700** | 0.567 | 0.686 |
| Deco_hop                  | 0.602    | **0.616** | 0.605 | 0.586 | 0.610 |
| Fexofenadine_MPO          | 0.668    | 0.680 | 0.663 | 0.673 | **0.699** |
| Median_molecules_1        | 0.199    | 0.197 | 0.321 | 0.330 | **0.346** |
| Median_molecules_2        | 0.195    | 0.208 | 0.224 | 0.218 | **0.260** |
| Mestranol_similarity      | 0.454    | 0.514 | 0.645 | 0.657 | **0.732** |
| Osimertinib_MPO           | 0.782    | 0.791 | 0.780 | 0.778 | **0.793** |
| Perindopril_MPO           | 0.430    | 0.431 | 0.444 | 0.462 | **0.477** |
| QED                       | 0.922    | 0.925 | 0.927 | 0.928 | **0.930** |
| Ranolazine_MPO            | 0.626    | 0.635 | 0.681 | 0.655 | **0.663** |
| Scaffold_hop              | 0.758    | **0.772** | 0.764 | 0.735 | 0.754 |
| Sitagliptin_MPO           | 0.226    | 0.219 | 0.272 | **0.298** | 0.191 |
| Thiothixene_rediscovery   | 0.350    | 0.385 | 0.446 | 0.456 | **0.498** |
| Troglitazone_rediscovery  | 0.256    | 0.282 | 0.305 | 0.363 | **0.432** |
| Valsartan_smarts          | 0.012    | 0.011 | 0.010 | 0.010 | **0.015** |
| Zaleplon_MPO              | 0.408    | 0.412 | 0.415 | 0.401 | **0.432** |
| DRD2                      | 0.907    | 0.906 | 0.942 | **0.944** | 0.930 |
| GSK3B                     | 0.738    | 0.719 | 0.781 | 0.843 | **0.853** |
| JNK3                      | 0.640    | 0.649 | 0.660 | 0.725 | **0.744** |
| Total                     | 11.985   | 12.205| 13.203| 13.561| **14.037** |
