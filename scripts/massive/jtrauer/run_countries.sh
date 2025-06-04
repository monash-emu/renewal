#!/bin/bash


#SBATCH --job-name=arrayjob_renewal
#SBATCH --account=sh30

# The maximum allowed time for a job to run
#SBATCH --time=03:00:00

# 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096

# Number of CPUs for each (virtual) machine
# Set this to the number of cores you need;
# e.g. for 4 chains, use 4 CPUs
#SBATCH --cpus-per-task=8

# To receive an email when job completes or fails
# Delete this section if you don't require email notifications
#SBATCH --mail-user=james.trauer@monash.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Set this to the range of values that will be received in your script
# e.g. --array=1-4
# will result in 4 jobs, receiving 1, 2, 3, 4 respectively

#SBATCH --array=1-4

# The path to your renewal repo
cd /projects/sh30/users/jtrauer/renewal

# The run path to your Python script that will actually do the work
# Run this with pixi, the environment will be automatically handled
pixi run python scripts/massive/jtrauer/run_countries.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID