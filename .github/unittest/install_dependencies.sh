python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov hydra-core tqdm torch torchvision
python -m pip install rdkit==2023.3.3
python -m pip MolScore promptsmiles

if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

pip install torchrl

cd ../acegen-open
pip install -e .
