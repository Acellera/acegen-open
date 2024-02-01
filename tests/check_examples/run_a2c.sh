#!/bin/bash

#SBATCH --job-name=a2c
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/a2c%j.txt
#SBATCH --error=slurm_errors/a2c%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="acegen-open-example-check-$current_commit"
agent_name="a2c"

export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/examples/a2c/a2c.py \
  logger_backend=wandb \
  experiment_name="$project_name" \
  agent_name="$agent_name" \
  molscore=MolOpt.Albuterol_similarity

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${group_name}_${SLURM_JOB_ID}=success" >>> report.log
else
  echo "${group_name}_${SLURM_JOB_ID}=error" >>> report.log
fi
