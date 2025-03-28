# COMEBERT - Incorporating [DNABERT](https://arxiv.org/abs/2402.08777) embeddings into the [COMEBin](https://github.com/ziyewang/COMEBin) metagenomics binning pipeline

## How it works - Multiview contrastive learning

1. Comebin extracts sequencing depth information from BAM files mapping reads to contigs
2. Augmentations are made for each read (by taking subsequences)
3. DNABERT generates embeddings for each read
4. The sequencing depth information is sent through a network and combined with DNABERT embeddings
5. The combined embeddings from all augmentations are used to train a contrastive learning model
6. The original reads are sent through the contrastive model and assigned to bins

## Running the program

1. Create a conda environment from the `comebin_dnabert_env.yaml` file:

   `conda env create --file comebin_dnabert_env.yml`
   
   then install torch separately with pip:
   
   `pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113`
3. If you dont already have BAM files mapping reads to contigs, modify and run the script in `scripts/run_gen_cov.slurm` to with your reads and contigs to generate BAM files for input
4. Add your contigs, bamfiles, and output to `run_comebin.slurm` and run
5. The pipeline will complete with a text file in CAMI format mapping contigs to bins

## Evaluation

After evaluating the enhanced model on CAMI2 marine dataset, it performs better than out run of COMEBin base (with TNF) with the same parameters, but worse than the published results of COMEBin base (although the parameters used for evaluation were not published by the authors)

Results are shown on the [CAMI2 challenge website](https://cami-challenge.org/datasets/Marine/marmgCAMI2_short_read_pooled_gold_standard_assembly/genome_binning/)
