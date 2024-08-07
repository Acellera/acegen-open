#!/bin/bash

set -e

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install flake8 pytest pytest-cov hydra-core tqdm

# Install rdkit separately
python -m pip install rdkit==2023.3.3

# Verify rdkit installation
python -c "import rdkit; print('rdkit:', rdkit.__version__)"

# Not using nightly torch
python -m pip install torch torchvision
# python -m pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall

# Install additional packages
python -m pip install transformers promptsmiles torchrl MolScore

# Install new dependencies
python -m pip install causal-conv1d>=1.4.0 mamba-ssm

# Verify installations
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import promptsmiles; print('promptsmiles:', promptsmiles.__version__)"
python -c "import mamba; print('mamba-ssm:', mamba.__version__)"
python -c "from rdkit.Chem import AllChem as Chem; print('rdkit AllChem imported successfully')"

# Print Python path and installed packages
python -c "import sys; print('Python path:', sys.path)"
python -m pip list
cd ../acegen-open
pip install -e .
pip uninstall --yes torchrl
pip uninstall --yes tensordict

cd ..
python -m pip install git+https://github.com/pytorch-labs/tensordict.git
git clone https://github.com/pytorch/rl.git
cd rl
python setup.py develop
