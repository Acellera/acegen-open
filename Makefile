CONDA ?= conda
ENV_STABLE  = acegen-stable
ENV_NIGHTLY = acegen-nightly

# ── Stable stack (mirrors .github/unittest/install_dependencies.sh) ──────────
.PHONY: setup-stable
setup-stable:
	$(CONDA) create -n $(ENV_STABLE) python=3.10 ninja cmake -y
	$(CONDA) run --no-capture-output -n $(ENV_STABLE) pip install --upgrade pip
	$(CONDA) run --no-capture-output -n $(ENV_STABLE) pip install \
	    flake8 pytest pytest-cov hydra-core tqdm packaging
	$(CONDA) run --no-capture-output -n $(ENV_STABLE) pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128 
	$(CONDA) run --no-capture-output -n $(ENV_STABLE) pip install tensordict==0.11.0 torchrl==0.11.1
	$(CONDA) run --no-capture-output -n $(ENV_STABLE) pip install \
	    transformers promptsmiles MolScore \
	    deepsmiles selfies atomInSmiles safe-mol smizip molbloom wheel
	$(CONDA) run --no-capture-output -n $(ENV_STABLE) pip install -e .

.PHONY: test-stable
test-stable:
	$(CONDA) run --no-capture-output -n $(ENV_STABLE) pytest tests/ --cov=acegen

# ── Nightly stack (mirrors .github/unittest/install_dependencies_nightly.sh) ─
.PHONY: setup-nightly
setup-nightly:
	$(CONDA) create -n $(ENV_NIGHTLY) python=3.10 ninja cmake -y
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install --upgrade pip
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install \
	    flake8 pytest pytest-cov hydra-core tqdm packaging
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install torchrl tensordict
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install \
	    transformers promptsmiles torchrl rdkit MolScore
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install \
	    deepsmiles selfies atomInSmiles safe-mol smizip molbloom wheel
#	$(CONDA) install -n $(ENV_STABLE) \
		-c nvidia/label/cuda-12.4.0 cuda-nvcc -y
#	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install \
	    causal-conv1d>=1.4.0 --no-build-isolation
#	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install \
	    mamba-ssm==1.2.2 --no-build-isolation
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pip install -e .

.PHONY: test-nightly
test-nightly:
	$(CONDA) run --no-capture-output -n $(ENV_NIGHTLY) pytest tests/ --cov=acegen

# ── Current active environment ────────────────────────────────────────────────
.PHONY: test
test:
	pytest tests/ --cov=acegen

# ── Clean up environments ─────────────────────────────────────────────────────
.PHONY: clean-envs
clean-envs:
	$(CONDA) env remove -n $(ENV_STABLE) -y || true
	$(CONDA) env remove -n $(ENV_NIGHTLY) -y || true
