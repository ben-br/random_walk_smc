#!/bin/bash
#SBATCH --mail-user=bloemred@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=test
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 
#cores required for each job
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=4gb
#SBATCH --array=1-2
#SBATCH --output=/data/localhost/not-backed-up/bloemred/random_walk_smc/output/test_%A_%a_out.txt

# run Julia script
julia test.jl

# print environment variables: the job ID, sub-job's task ID, and sub-job’s job ID.
echo "SLURM_JOB_NAME: " $SLURM_JOB_NAME
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Make a directory for output (.txt) and results (e.g. .jld, .json ,.mat) if it doesn't already exist
mkdir -p /data/ziz/not-backed-up/bloemred/outputs/test_${SLURM_ARRAY_JOB_ID}
mkdir -p /data/ziz/not-backed-up/bloemred/results/test_${SLURM_ARRAY_JOB_ID}

# Move experiment outputs & results to the directories made above
mv /data/localhost/not-backed-up/bloemred/test_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt /data/ziz/not-backed-up/username/bloemred/test_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.txt
mv /data/localhost/not-backed-up/bloemred/test_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /data/ziz/not-backed-up/bloemred/results/test_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.out
