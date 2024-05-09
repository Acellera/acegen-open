#!/bin/bash

# Function to display script usage
display_usage() {
    cat <<EOF
Usage: ./run-example-scripts.sh [OPTIONS]

OPTIONS:
  --partition PARTITION   Specify the Slurm partition for the job.
  --n_runs N_RUNS         Specify the number of runs for each script. Default is 1.

EXAMPLES:
  ./run-example-scripts.sh --partition <PARTITION_NAME> --n_runs 5

EOF
    return 1
}

# Check if the script is called with --help or without any arguments
if [ "$1" == "--help" ]; then
    display_usage
    exit 1
fi

# Initialize variables with default values
n_runs="1"
slurm_partition=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --n_runs)
      n_runs="$2"
      shift 2
      ;;
    --partition)
      slurm_partition="$2"
      shift 2
      ;;
    *)
      echo "$1 is not a valid argument. See './run-example-scripts.sh --help'."
      return 0
      ;;
  esac
done

scripts=(
    run_pretrain_single_node.sh
    run_pretrain_distributed.sh

    run_reinforce_denovo.sh
    run_reinvent_denovo.sh
    run_ahc_denovo.sh
    run_a2c_denovo.sh
    run_ppo_denovo.sh
    run_ppod_denovo.sh

    run_reinforce_scaffold.sh
    run_reinvent_scaffold.sh
    run_ahc_scaffold.sh
    run_a2c_scaffold.sh
    run_ppo_scaffold.sh
    run_ppod_scaffold.sh

    run_reinforce_fragment.sh
    run_reinvent_fragment.sh
    run_ahc_fragment.sh
    run_a2c_fragment.sh
    run_ppo_fragment.sh
    run_ppod_fragment.sh
)

# Remove the previous logs and errors
rm -rf "slurm_errors"
rm -rf "slurm_logs"

# Create the logs and errors directories
mkdir -p "slurm_errors"
mkdir -p "slurm_logs"

# Remove the previous report
rm -f report.log

# Submit jobs with the specified partition the specified number of times
if [ -z "$slurm_partition" ]; then
    for script in "${scripts[@]}"; do
        for ((i=1; i<=$n_runs; i++)); do
            export N_RUN=$i
            sbatch "$script"
        done
    done
else
  for script in "${scripts[@]}"; do
      for ((i=1; i<=$n_runs; i++)); do
          export N_RUN=$i
          sbatch --partition="$slurm_partition" "$script"
      done
  done
fi
