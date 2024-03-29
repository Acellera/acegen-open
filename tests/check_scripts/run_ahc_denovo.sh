#!/bin/bash

#SBATCH --job-name=ahc_denovo
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/ahc_denovo%j.txt
#SBATCH --error=slurm_errors/ahc_denovo%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="acegen-scripts-check-$current_commit"
agent_name="ahc_denovo"

export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/scripts/ahc/ahc.py \
  logger_backend=wandb \
  experiment_name="$project_name" \
  agent_name="$agent_name" \
  molscore=MolOpt \
  molscore_include=[Albuterol_similarity] \
  seed=$N_RUN \
  log_dir="$agent_name"_seed"$N_RUN"

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${agent_name}_${SLURM_JOB_ID}=success" >> report.log
else
  echo "${agent_name}_${SLURM_JOB_ID}=error" >> report.log
fi

mv "$agent_name"_seed"$N_RUN"* slurm_logs/
