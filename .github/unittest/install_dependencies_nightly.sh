# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install flake8 pytest pytest-cov hydra-core tqdm
python -m pip install torch torchvision
python -m pip install transformers promptsmiles torchrl rdkit==2023.3.3 MolScore # causal-conv1d>=1.4.0 mamba-ssm==1.2.2
pip install deepsmiles selfies smi2sdf smi2svg # atomInSmiles safe

# Verify installations
python -c "import transformers; print(transformers.__version__)"
python -c "import promptsmiles"
# python -c "import mamba_ssm; print('mamba-ssm:', mamba_ssm.__version__)"  # Assuming mamba-ssm imports as mamba

# Install local package
cd ../acegen-open
pip install -e .
pip uninstall --yes torchrl
pip uninstall --yes tensordict

# Install torchrl and tensordict nightly
cd ..
python -m pip install git+https://github.com/pytorch-labs/tensordict.git
git clone https://github.com/pytorch/rl.git
cd rl
python setup.py develop
