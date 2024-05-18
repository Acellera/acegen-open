
<p align="center">
  <img src="./acegen/images/acegen_logo.jpeg" alt="Alt Text" width="250" />
</p>

# AceGen: A TorchRL-based toolkit for reinforcement learning in generative chemistry

---

## Overview

AceGen is a comprehensive toolkit designed to leverage reinforcement learning (RL) techniques for generative chemistry tasks, particularly in drug design. AceGen harnesses the capabilities of TorchRL, a modern library for general decision-making tasks, to provide a flexible and integrated solution for generative drug design challenges.

![Alt Text](./acegen/images/train_zaleplon.png)

The full paper can be found [here](https://arxiv.org/abs/2405.04657).

---

## Features

- **Multiple Generative Modes:** AceGen facilitates the generation of chemical libraries with different modes: de novo generation, scaffold decoration, and fragment linking.
- **RL Algorithms:** AceGen offers task optimization with various reinforcement learning algorithms such as [Proximal Policy Optimization (PPO)][1], [Advantage Actor-Critic (A2C)][2], [Reinforce][3], [Reinvent][4], and [Augmented Hill-Climb (AHC)][5].
- **Pre-trained Models:** The toolkit offers pre-trained models including Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), and GPT-2.
- **Scoring Functions :** AceGen relies on MolScore, a comprehensive scoring function suite for generative chemistry, to evaluate the quality of the generated molecules.
- **Customization Support:** AceGen provides tutorials for integrating custom models and custom scoring functions, ensuring flexibility for advanced users.

---

![Alt Text](./acegen/images/chem_zaleplon.png)

---

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

To install AceGen, run (use `pip install -e ./` for develop mode)

    pip3 install tqdm wandb hydra-core
    git clone https://github.com/Acellera/acegen-open.git
    cd acegen-open
    pip install ./

### Optional dependencies

Unless you intend to define your own custom scoring functions, install MolScore by running

    pip3 install MolScore

To use the scaffold decoration and fragment linking, install promptsmiles by running

    pip3 install promptsmiles

To learn how to configure constrained molecule generation with AcGen and promptsmiles, please refer to this [tutorial](tutorials/using_promptsmiles.md).

---

## Generating libraries of molecules

ACEGEN has multiple RL algorithms available, each in a different directory within the `acegen-open/scripts` directory. Each RL algorithm has three different generative modes of execution: de novo, scaffold decoration, and fragment linking.

Each mode of execution has its own configuration file in YAML format, located right next to the script. To modify training parameters for any mode, edit the corresponding YAML file. For a breakdown of the general structure of our configuration files, refer to this [tutorial](tutorials/breaking_down_configuration_files.md).

While the default values in the configuration files are considered sensible, a default scoring function and model architecture are also defined so users can test the scripts out of the box. However, users might generally want to customize the model architecture or the scoring function.

To customize the model architecture, refer to the [Changing the model architecture](##Changing the model architecture) section. To customize the scoring function, refer to the [Changing the scoring function](##Changing the scoring function) section.

### Running training scripts to generate compoud libraries

To run the training scripts for denovo generation, run the following commands:
    
    python scripts/reinforce/reinforce.py --config-name config_denovo
    python scripts/a2c/a2c.py --config-name config_denovo
    python scripts/ppo/ppo.py --config-name config_denovo
    python scripts/reinvent/reinvent.py --config-name config_denovo
    python scripts/ahc/ahc.py --config-name config_denovo

To run the training scripts for scaffold decoration, run the following commands (requires installation of promptsmiles):

    python scripts/reinforce/reinforce.py --config-name config_scaffold
    python scripts/a2c/a2c.py --config-name config_scaffold
    python scripts/ppo/ppo.py --config-name config_scaffold
    python scripts/reinvent/reinvent.py --config-name config_scaffold
    python scripts/ahc/ahc.py --config-name config_scaffold

To run the training scripts for fragment linking, run the following commands (requires installation of promptsmiles):

    python scripts/reinforce/reinforce.py --config-name config_linking
    python scripts/a2c/a2c.py --config-name config_linking
    python scripts/ppo/ppo.py --config-name config_linking
    python scripts/reinvent/reinvent.py --config-name config_linking
    python scripts/ahc/ahc.py --config-name config_linking

#### Advanced usage

Scripts are also available as executables after installation, but both the path and name of the config must be specified. For example,

    ppo.py --config-path=<path_to_config_dir> --config-name=<config_name.yaml> 

YAML config parameters can also be specified on the command line. For example,

    ppo.py --config-path=<path_to_config_dir> --config-name=<config_name.yaml> total_smiles=100

---

## Changing the scoring function

To change the scoring function, adjust the `molscore` parameter in any configuration files. Set it to point to a valid 
MolScore configuration file (e.g.  ../MolScore/molscore/configs/GuacaMol/Albuterol_similarity.json). 
Alternatively, you can set the `molscore` parameter to the name of a valid MolScore benchmark 
(such as MolOpt, GuacaMol, etc.) to automatically execute each task in the benchmark. For further details on MolScore, 
please refer to the [MolScore](https://github.com/MorganCThomas/MolScore) repository.

Alternatively, users can define their own custom scoring functions and use them in the AceGen scripts by following the 
instructions in this [tutorial](tutorials/adding_custom_scoring_function.md).

---

## Changing the model architecture

### Available models

We provide a variety of default priors that can be selected in the configuration file. These include:

- A Gated Recurrent Unit (GRU) model
  - pre-training dataset1 (default): [ChEMBL](https://www.ebi.ac.uk/chembl/)
  - pre-training dataset2: [ZINC250k](https://github.com/wenhao-gao/mol_opt/blob/main/data/zinc.txt.gz)
  - number of parameters: 4,363,045
  - to select set the field `model` to `gru` in any configuration file


- A Long Short-Term Memory (LSTM) model
  - pre-training dataset: [ChEMBL](https://www.ebi.ac.uk/chembl/)
  - number of parameters: 5,807,909
    - to select set the field `model` to `lstm` in any configuration file
 

- A GPT-2 model (requires installation of HuggingFace's `transformers` library)
  - pre-training dataset: [REAL 350/3 lead-like, 613.86M cpds, CXSMILES](https://enamine.net/compound-collections/real-compounds/real-database-subsets)
  - number of parameters: 5,030,400
- to select set the field `model` to `gpt2` in any configuration file

---

### Integration of custom models

We also encourage users to integrate their own models into AceGen.

A detailed guide on integrating custom models can be found in this [tutorial](tutorials/adding_custom_model.md).

---

## Results on the [MolOpt](https://arxiv.org/pdf/2206.12411.pdf) benchmark

Algorithm comparison for the Area Under the Curve (AUC) of the top 100 molecules on MolOpt benchmark scoring functions. 
Each algorithm ran 5 times with different seeds, and results were averaged. 
We used the default configuration for each algorithm, including the GRU model for the prior.
Additionally, for Reinvent we also tested the configuration proposed in the MolOpt paper.

| Task                          | [REINFORCE][3] | [REINVENT][4] | [REINVENT MolOpt][6] | [AHC][5]   | [A2C][2]   | [PPO][1]   | [PPOD][7]  |
|-------------------------------|-----------|----------|-----------------|-------|-------|-------|-------|-------|
| Albuterol_similarity | 0.68 ± 0.03 | 0.69 ± 0.02 | 0.90 ± 0.01    | 0.77 ± 0.02 | 0.82 ± 0.04 | 0.93 ± 0.02 | **0.94 ± 0.00** |
| Amlodipine_MPO | 0.55 ± 0.01 | 0.56 ± 0.01 | 0.65 ± 0.06      | 0.56 ± 0.01 | 0.55 ± 0.01 | 0.58 ± 0.03 | **0.68 ± 0.02** |
| C7H8N2O2  | 0.83 ± 0.01 | 0.82 ± 0.03 | **0.90 ± 0.02**   | 0.76 ± 0.04 | 0.84 ± 0.04 | 0.89 ± 0.01 | 0.89 ± 0.03 |
| C9H10N2O2PF2Cl | 0.70 ± 0.02 | 0.70 ± 0.02 | 0.76 ± 0.03    | 0.68 ± 0.02 | 0.69 ± 0.03 | 0.66 ± 0.02 | **0.79 ± 0.02** |
| Celecoxxib_rediscovery | 0.63 ± 0.02 | 0.64 ± 0.03 | 0.77 ± 0.02 | 0.72 ± 0.02 | 0.73 ± 0.06 | 0.65 ± 0.12 | **0.82 ± 0.03** |
| DRD2     | 0.98 ± 0.00 | 0.97 ± 0.00 | **0.99 ± 0.00** | 0.98 ± 0.01 | 0.98 ± 0.01 | **0.99 ± 0.00** | **0.99 ± 0.00** |
| Deco_hop             | 0.63 ± 0.00    | 0.63 ± 0.01    | **0.67 ± 0.01**   | 0.64 ± 0.01    | 0.62 ± 0.00    | 0.62 ± 0.01    | 0.66 ± 0.02    |
| Fexofenadine_MPO     | 0.71 ± 0.01    | 0.71 ± 0.00    | **0.80 ± 0.03**   | 0.72 ± 0.00    | 0.71 ± 0.00    | 0.73 ± 0.00    | 0.78 ± 0.01    |
| GSK3B                | 0.84 ± 0.01    | 0.84 ± 0.02    | **0.92 ± 0.02**   | 0.82 ± 0.01    | 0.85 ± 0.02    | 0.90 ± 0.02    | **0.92 ± 0.02**|
| JNK3                 | 0.75 ± 0.03    | 0.75 ± 0.02    | 0.85 ± 0.04       | 0.75 ± 0.01    | 0.74 ± 0.06    | 0.80 ± 0.04    | **0.87 ± 0.02**|
| Median_molecules_1   | 0.26 ± 0.00    | 0.24 ± 0.00    | **0.36 ± 0.02**   | 0.24 ± 0.00    | 0.31 ± 0.01    | 0.33 ± 0.02    | 0.35 ± 0.02    |
| Median_molecules_2   | 0.22 ± 0.00    | 0.22 ± 0.00    | 0.28 ± 0.01       | 0.24 ± 0.00    | 0.25 ± 0.01    | 0.25 ± 0.02    | **0.29 ± 0.01**|
| Mestranol_similarity | 0.60 ± 0.03    | 0.55 ± 0.04    | 0.85 ± 0.07       | 0.66 ± 0.04    | 0.69 ± 0.07    | 0.75 ± 0.15    | **0.89 ± 0.05**|
| Osimertinib_MPO      | 0.82 ± 0.01    | 0.82 ± 0.00    | **0.86 ± 0.01**   | 0.83 ± 0.00    | 0.81 ± 0.01    | 0.82 ± 0.01    | 0.84 ± 0.00    |
| Perindopril_MPO      | 0.48 ± 0.01    | 0.47 ± 0.00    | **0.54 ± 0.01**   | 0.47 ± 0.01    | 0.48 ± 0.00    | 0.50 ± 0.01    | 0.53 ± 0.00    |
| QED                  | **0.94 ± 0.00**| **0.94 ± 0.00**| **0.94 ± 0.00**   | **0.94 ± 0.00**| **0.94 ± 0.00**| **0.94 ± 0.00**| **0.94 ± 0.00**|
| Ranolazine_MPO        | 0.70 ± 0.01    | 0.69 ± 0.00    | **0.76 ± 0.01**   | 0.70 ± 0.00    | 0.74 ± 0.01    | 0.73 ± 0.01    | 0.75 ± 0.00    |
| Scaffold_hop              | 0.80 ± 0.00    | 0.79 ± 0.00    | **0.86 ± 0.02**   | 0.80 ± 0.01    | 0.80 ± 0.00    | 0.80 ± 0.02    | 0.84 ± 0.03    |
| Sitagliptin_MPO           | 0.34 ± 0.02    | 0.33 ± 0.01    | 0.38 ± 0.03       | 0.33 ± 0.02    | **0.39 ± 0.02**| 0.32 ± 0.02    | **0.39 ± 0.02**|
| Thiothixene_rediscovery   | 0.41 ± 0.01    | 0.41 ± 0.00    | 0.56 ± 0.04       | 0.45 ± 0.02    | 0.48 ± 0.04    | 0.48 ± 0.06    | **0.58 ± 0.09**|
| Troglitazone_rediscovery  | 0.31 ± 0.02    | 0.31 ± 0.02    | 0.47 ± 0.05       | 0.34 ± 0.01    | 0.35 ± 0.02    | 0.46 ± 0.07    | **0.52 ± 0.06**|
| Valsartan_smarts          | **0.03 ± 0.00**| 0.02 ± 0.00    | 0.02 ± 0.00       | 0.02 ± 0.00    | 0.02 ± 0.00    | **0.03 ± 0.00**| **0.03 ± 0.00**|
| Zaleplon_MPO              | 0.47 ± 0.01    | 0.47 ± 0.01    | **0.52 ± 0.01**   | 0.48 ± 0.01    | 0.47 ± 0.01    | 0.50 ± 0.02    | **0.52 ± 0.01**|
| **Total**                 | **13.67**      | **13.60**      | **15.65**         | **13.91**      | **14.27**      | **14.65**      | **15.80**      |


[1]: https://arxiv.org/abs/1707.06347
[2]: https://arxiv.org/abs/1602.01783
[3]: https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
[4]: https://arxiv.org/abs/1704.07555
[5]: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00646-z
[6]: https://arxiv.org/pdf/2206.12411.pdf
[7]: https://arxiv.org/abs/2007.03328
