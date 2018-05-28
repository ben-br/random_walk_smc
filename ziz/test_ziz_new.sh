#!/bin/bash
#note anything after #SBATCH is a command
#SBATCH --mail-user=bloemred@stats.ox.ac.uk
#Email you if job starts, completed or failed
#SBATCH --mail-type=ALL
#SBATCH --job-name=sample_job
#SBATCH --partition=small 
#Choose your partition depending on your requirements
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#Cpus required for each job
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=100
#Memory per cpu in megabytes          
#SBATCH --array=1-10    
#SBATCH --output=/data/localhost/not-backed-up/bloemred/%x_%A_%a.out
#SBATCH --error=/data/localhost/not-backed-up/bloemred/%x_%A_%a.err

julia test.jl

# print environment variables: the job ID, sub-job's task ID, and sub-job’s job ID.
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Make a directory for output (.txt) and results (e.g. .jld, .json ,.mat) if it doesn't already exist
mkdir -p /data/ziz/not-backed-up/bloemred/outputs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
mkdir -p /data/ziz/not-backed-up/bloemred/results/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}

# Move experiment outputs & results to the directories made above
mv /data/localhost/not-backed-up/bloemred/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err /data/ziz/not-backed-up/bloemred/outputs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err
mv /data/localhost/not-backed-up/bloemred/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /data/ziz/not-backed-up/bloemred/outputs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
