#!/bin/bash
#SBATCH --mail-user=bloemred@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=prior_sensitivity_100
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --array=1-2
#SBATCH --output=/data/localhost/not-backed-up/bloemred/%x_%A_%a.out
#SBATCH --error=/data/localhost/not-backed-up/bloemred/%x_%A_%a.err

#mkdir -p /data/ziz/not-backed-up/bloemred/random_walk_smc/outputs # for output files
# Make a directory for output (.txt) and results (e.g. .jld, .json ,.mat) if it doesn't already exist
mkdir -p /data/ziz/not-backed-up/bloemred/random_walk_smc/results/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}

# Run Julia code
julia prior_test_ziz_100.jl

# print environment variables: the job ID, sub-job's task ID, and sub-job’s job ID.
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


# Move experiment outputs & results
mv /data/localhost/not-backed-up/bloemred/prior_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jld /data/ziz/not-backed-up/bloemred/random_walk_smc/results/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/prior_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jld
mv /data/localhost/not-backed-up/bloemred/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /data/ziz/not-backed-up/bloemred/random_walk_smc/outputs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
mv /data/localhost/not-backed-up/bloemred/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err /data/ziz/not-backed-up/bloemred/random_walk_smc/outputs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err



