#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -J fp_parallel
#SBATCH -A desi
#SBATCH -c 1                 # reserve 3 CPUs on this node
#SBATCH -o fp_parallel.%j.out

# Environment
source /global/common/software/desi/desi_environment.sh main
conda activate desi

export OMP_NUM_THREADS=1      # avoid thread oversubscription in libs
export MPLBACKEND=Agg         # headless matplotlib

CHUNK_DIR=/pscratch/sd/j/jlargett/DESI_SGA_MINE/Sorter/FP/chunks_missing_4
OUT=/pscratch/sd/j/jlargett/DESI_SGA_MINE/Sorter/FP/fpy3_cutouts
mkdir -p "$OUT" logs

# Choose a polite level of parallelism for the remote server:
J=${SLURM_CPUS_PER_TASK:-1}  # match -c 1 above

# Run: one Python process per chunk CSV
parallel --jobs ${J} \
         --delay 0.10 \
         --joblog logs/parallel_joblog.tsv \
         --results logs/parallel_results \
         --eta --bar \
  'python FP_cutouts.py \
      --csv {1} \
      --outdir '"$OUT"' \
      --bands grz --size 152 \
      --timeout 20 --retries 3 --sleep-ms 50' \
  ::: "$CHUNK_DIR"/chunk_*.csv