# COMEBERT - Incorporating [DNABERT](https://arxiv.org/abs/2402.08777) embeddings into the [COMEBin](https://github.com/ziyewang/COMEBin) metagenomics binning pipeline

## How it works - Multiview contrastive learning

1. Comebin extracts sequencing depth information from BAM files mapping reads to contigs
2. Augmentations are made for each read (by taking subsequences)
3. DNABERT generates embeddings for each read
4. The sequencing depth information is sent through a network and combined with DNABERT embeddings
5. The combined embeddings from all augmentations are used to train a contrastive learning model
6. The original reads are sent through the contrastive model and assigned to bins
