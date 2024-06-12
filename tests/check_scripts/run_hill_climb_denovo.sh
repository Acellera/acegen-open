#!/bin/bash

#SBATCH --job-name=hill_climb_denovo
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/hill_climb_denovo%j.txt
#SBATCH --error=slurm_errors/hill_climb_denovo%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="acegen-scripts-check-$current_commit"
agent_name="hill_climb_denovo"
if [ -z "$N_RUN" ]; then
  echo "N_RUN is not set. Setting to default value of 1."
  N_RUN=1
fi
if [ -z "$ACEGEN_MODEL" ]; then
  echo "ACEGEN_MODEL is not set. Setting to default value of gru. Choose from [gru, lstm, gpt2]"
  ACEGEN_MODEL="gru"
fi

export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/scripts/hill_climb/hill_climb.py \
  logger_backend=wandb \
  experiment_name="$project_name" \
  agent_name="$agent_name" \
  seed=$N_RUN \
  log_dir="$agent_name"_seed"$N_RUN" \
  model=$ACEGEN_MODEL

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${agent_name}_${SLURM_JOB_ID}=success" >> report.log
else
  echo "${agent_name}_${SLURM_JOB_ID}=error" >> report.log
fi

mv "$agent_name"_seed"$N_RUN"* slurm_logs/
