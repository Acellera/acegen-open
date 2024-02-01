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
    # run_a2c.sh
    # run_ppo.sh
    run_reinvent.sh
    #run_ahc.sh
)

mkdir -p "slurm_errors"
mkdir -p "slurm_logs"

# remove the previous report
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
