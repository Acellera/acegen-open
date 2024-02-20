#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/pretrain%j.txt
#SBATCH --error=slurm_errors/pretrain%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="acegen-open-example-check-$current_commit"
agent_name="pretrain"

export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/examples/pretrain/pretrain_single_node.py \
  logger_backend=wandb \
  experiment_name="$project_name" \
  agent_name="$agent_name" \
  seed=$N_RUN \
  model_log_dir="$agent_name"_seed"$N_RUN" \
  dataset_log_dir="$agent_name"_seed"$N_RUN" \
  epochs=10 \
  train_dataset_path=$PYTHONPATH/tests/data/smiles_test_set

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${agent_name}_${SLURM_JOB_ID}=success" >> report.log
else
  echo "${agent_name}_${SLURM_JOB_ID}=error" >> report.log
fi

mv "$agent_name"_seed"$N_RUN"* slurm_logs/
