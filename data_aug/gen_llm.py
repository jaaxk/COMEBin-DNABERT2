#!/usr/bin/env python

import os
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as util_data
from tqdm import tqdm
from sklearn.preprocessing import normalize
import csv
from Bio import SeqIO

def run_gen_dnabert(fasta_file, args):
    outfile = os.path.join(os.path.dirname(fasta_file), f'{args.model_name}_embeddings.csv')
    
    # Determine model path
    if args.llm_model_path is None:
        model_name_or_path = {
            'dnabert2': "zhihan1996/DNABERT-2-117M",
            'dnabert-s': "zhihan1996/DNABERT-S"
        }.get(args.model_name)
    else:
        model_name_or_path = args.llm_model_path

    # Load model and tokenizer once, outside of worker initialization
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side='right',
        use_fast=True,
        trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Read sequences
    sequences = [(str(record.id), str(record.seq)) 
                 for record in SeqIO.parse(fasta_file, 'fasta')]
    
    # Sort by length for efficient batching
    sequences.sort(key=lambda x: len(x[1]))
    contig_ids, dna_sequences = zip(*sequences)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()  # Explicitly set to evaluation mode

    # Create DataLoader
    train_loader = util_data.DataLoader(
        dna_sequences, 
        batch_size=args.llm_batch_size * (torch.cuda.device_count() or 1),
        shuffle=False,
        num_workers=2 * (torch.cuda.device_count() or 1)
    )

    embeddings = []
    
    for batch in tqdm(train_loader, desc='Generating embeddings'):
        with torch.no_grad():
            # Tokenize
            token_feat = tokenizer.batch_encode_plus(
                batch,
                max_length=args.model_max_length,
                return_tensors='pt',
                padding='longest',
                truncation=True
            )
            
            # Move to device
            input_ids = token_feat['input_ids'].to(device)
            attention_mask = token_feat['attention_mask'].to(device)
            
            # Get embeddings
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1)
            token_embeddings = outputs[0] * mask
            
            # Sum and normalize
            seq_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeddings.append(seq_embeddings.detach().cpu())

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = normalize(embeddings.numpy())

    # Write to CSV
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['contig_id'] + [f'dim_{i}' for i in range(embeddings.shape[1])]
        writer.writerow(header)
        
        for contig_id, embedding in tqdm(zip(contig_ids, embeddings), 
                                       total=len(contig_ids), 
                                       desc="Writing to CSV"):
            writer.writerow([contig_id] + embedding.tolist())

    return outfile
