# Tutorial: Breaking down configuration files

---

In this tutorial, we will walk through the structure of configuration files used in our ACEGEN training scripts. All our configuration files follow a similar structure, making it easier to manage and understand the parameters for experiments with different modes and algorithms.

## Configuration File Structure

Here is the genral structure of our configuration file:

```yaml
# Logging configuration
...

# Environment configuration
...

# Scoring function
...

# Promptsmiles configuration
...

# Model architecture
...

# Optimizer configuration
...

# Algorithm configuration
...

# Data replay configuration
...
```

We will take the example `scripts/reinvent/config_denovo.yaml`, the config file for the novo generation using the REINVENT algorithm to break down each seaction.

## Logging configuration

```yaml
# Logging configuration
experiment_name: acegen
agent_name: reinvent
log_dir: results
logger_backend: null  # wandb, tensorboard, or null
seed: 101 # multiple seeds can be provided as a list to multiple experiments sequentially e.g. [101, 102, 103]
```

## Environment configuration

```yaml
# Environment configuration
num_envs: 128 # Number of smiles to generate in parallel
total_smiles: 10_000
```

## Scoring function

```yaml
# Scoring function
molscore: MolOpt
molscore_include: ["Albuterol_similarity"]
custom_task: null # Requires molscore to be set to null
```

## Promptsmiles configuration

```yaml
# Promptsmiles configuration
prompt: null  # e.g. c1ccccc
```

## Model architecture

```yaml
# Model architecture
model: gru # gru, lstm, or gpt2
# The default prior varies for each model. Refer to the README file in the root directory for more information.
# The default vocabulary varies for each prior. Refer to the README file in the root directory for more information.
custom_model_factory: null # Path to a custom model factory (e.g. my_module.create_model)
```

## Optimizer configuration

```yaml
# Optimizer configuration
lr: 0.0001
eps: 1.0e-08
weight_decay: 0.0
```

## Algorithm configuration

```yaml
# Reinvent configuration
sigma: 120
```

## Data replay configuration

```yaml
# Data replay configuration
experience_replay: True
replay_buffer_size: 100
replay_batch_size: 10
```