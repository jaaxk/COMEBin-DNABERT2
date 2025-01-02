#!/bin/bash
#
#SBATCH --job-name=gen_cov
#SBATCH --output=res.txt
#SBATCH --time=48:00:00
#SBATCH -p long-28core
#SBATCH --mem=120G
#SBATCH --cpus-per-task=28

cd /gpfs/scratch/jvaska/COMEBin/COMEBin-DNABERT2/scripts

conda init
conda activate comebin_env

chmod +x gen_cov_file.sh

export CONTIG_FILE=/gpfs/scratch/jvaska/COMEBin/data/marmg_gsa_pooled.fasta
export OUTPUT_DIR=/gpfs/scratch/jvaska/COMEBin/marmg_output/gen_cov_out
export READS=/gpfs/scratch/jvaska/COMEBin/data/reads/processed_files

./gen_cov_file.sh -a ${CONTIG_FILE} \
-o ${OUTPUT_DIR} \
--single-end \
-m 120 \
-t 28 \
${READS}/*fastq
