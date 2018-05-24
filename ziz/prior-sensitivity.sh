#!/bin/bash

#SBATCH --mail-user=bloemred@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=prior-sensitivity
#SBATCH --partition=medium
#Choose your partition depending on your requirements
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 #cores required for each job
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --array=1-2
#SBATCH --output=/data/localhost/not-backed-up/bloemred/random_walk_smc/output/prior-sensitivity_%A_%a_out.txt

mkdir -p /data/ziz/not-backed-up/bloemred/random_walk_smc/output # for output file
# Make a directory for output (.txt) and results (e.g. .jld, .json ,.mat) if it doesn't already exist
mkdir -p /data/ziz/not-backed-up/bloemred/random_walk_smc/results/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}

# Run Julia code
julia prior_test_ziz.jl

# print environment variables: the job ID, sub-job's task ID, and sub-job’s job ID.
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


# Move experiment outputs & results
mv /data/localhost/not-backed-up/bloemred/random_walk_smc/results/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/prior_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jld /data/ziz/not-backed-up/bloemred/random_walk_smc/results/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/prior_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jld
mv /data/localhost/not-backed-up/bloemred/random_walk_smc/output/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /data/ziz/not-backed-up/bloemred/random_walk_smc/output/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
