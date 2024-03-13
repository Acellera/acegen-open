# AceGen: A TorchRL-based toolkit for reinforcement learning in generative chemistry

## Overview

AceGen is a comprehensive toolkit designed to leverage reinforcement learning (RL) techniques for generative chemistry tasks, particularly in drug design. AceGen harnesses the capabilities of TorchRL, a modern library for general decision-making tasks, to provide a flexible and integrated solution for generative drug design challenges.

![Alt Text](./acegen/images/train_zaleplon.png)


## Features

- **Generative Modes:** AceGen facilitates the generation of chemical libraries with different modes: de novo generation, scaffold decoration, and fragment linking.
- **RL Algorithms:** AceGen offers task optimization with various reinforcement learning algorithms such as Proximal Policy Optimization (PPO), Advantage Actor-Critic (A2C), Reinvent, and Augmented Hill-Climb (AHC).
- **Pre-trained Models:** The toolkit offers pre-trained models including Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), and GPT-2.
- **Scoring Functions :** AceGen relies on MolScore, a comprehensive scoring function suite for generative chemistry, to evaluate the quality of the generated molecules.
- **Customization Support:** AceGen provides tutorials for integrating custom models and scoring functions, ensuring flexibility for advanced users.

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

### Install AceGen

To install AceGen, run

    pip3 install tqdm wandb hydra-core
    cd acegen-open
    python setup.py install

### Optional dependencies

Unless you intend to define your own custom scoring functions, install MolScore by running

    pip3 install MolScore

To use the scaffold decoration and fragment linking, install promptsmiles by running

    pip3 install promptsmiles

## Running training scripts

To run the training scripts for denovo generation, run the following commands:

    python scripts/a2c/a2c.py --config-name config_denovo
    python scripts/ppo/ppo.py --config-name config_denovo
    python scripts/reinvent/reinvent.py --config-name config_denovo
    python scripts/ahc/ahc.py --config-name config_denovo

To run the training scripts for scaffold decoration, run the following commands (requires installation of promptsmiles):

    python scripts/a2c/a2c.py --config-name config_scaffold
    python scripts/ppo/ppo.py --config-name config_scaffold
    python scripts/reinvent/reinvent.py --config-name config_scaffold
    python scripts/ahc/ahc.py --config-name config_scaffold

To run the training scripts for fragment linking, run the following commands (requires installation of promptsmiles):

    python scripts/a2c/a2c.py --config-name config_linking
    python scripts/ppo/ppo.py --config-name config_linking
    python scripts/reinvent/reinvent.py --config-name config_linking
    python scripts/ahc/ahc.py --config-name config_linking

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
 

- A GPT-2 model (requires installation of HuggingFace's `transformers` library)
  - pre-training dataset: [REAL 350/3 lead-like, 613.86M cpds, CXSMILES](https://enamine.net/compound-collections/real-compounds/real-database-subsets)
  - number of parameters: 5,030,400

# Changing the scoring function

To change the scoring function, adjust the `molscore` parameter in any configuration files. Set it to point to a valid 
MolScore configuration file (e.g.  ../MolScore/molscore/configs/GuacaMol/Albuterol_similarity.json). 
Alternatively, you can set the `molscore` parameter to the name of a valid MolScore benchmark 
(such as MolOpt, GuacaMol, etc.) to automatically execute each task in the benchmark. For further details on MolScore, 
please refer to the [MolScore](https://github.com/MorganCThomas/MolScore) repository.

Alternatively, users can define their own custom scoring functions and use them in the AceGen scripts by following the 
instructions in this [tutorial](tutorials/adding_custom_scoring_function.md).

# Integration of custom models

We encourage users to integrate their own models into AceGen.

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
