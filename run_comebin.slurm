#!/bin/bash
#
#SBATCH --job-name=s-comebin
#SBATCH --output=res-7.txt
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -p gpu-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28

cd /gpfs/scratch/jvaska/COMEBin/COMEBin-DNABERT2

export CONTIGS=/gpfs/scratch/jvaska/COMEBin/data/marmg_gsa_pooled.fasta
export BAMFILES=/gpfs/scratch/jvaska/COMEBin/marmg_output/gen_cov_out/bamfiles
export OUTPUT=/gpfs/scratch/jvaska/COMEBin/marmg_output_dnabert2_7aug

conda init
conda activate comebin_env
module load cuda113/toolkit/11.3

chmod +x run_comebin.sh

./run_comebin.sh -a ${CONTIGS} -o ${OUTPUT} -p ${BAMFILES} -t 28 -m dnabert2 -n 8
