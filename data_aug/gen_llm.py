#!/usr/bin/env python

import os
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig #For transformers v>4.28
from Bio import SeqIO
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as util_data
from tqdm import tqdm
from sklearn.preprocessing import normalize
import csv

def run_gen_dnabert(fasta_file, args):
    outfile = os.path.join(os.path.dirname(fasta_file), 'dnabert2_embeddings.csv')
    if args.llm_model_path is None:
        if args.model_name == 'dnabert2':
            model_name_or_path = "zhihan1996/DNABERT-2-117M"
    else:
        model_name_or_path = args.llm_model_path

    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    def init_tokenizer(worker_id):
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                    model_max_length = args.model_max_length,
                                                    padding_side = 'right',
                                                    use_fast = True, #Look into this
                                                    trust_remote_code=True)

    dna_sequences = []
    contig_ids = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        dna_sequences.append(str(record.seq))
        contig_ids.append(str(record.id))

    #Get embeddings
    #Sort input seqs by length for more efficient processing
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]

    #Use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device=='gpu':
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = nn.DataParallel(model)
            
    else:
        n_gpu = 1
    model.to(device)

    train_loader = util_data.DataLoader(dna_sequences, batch_size=args.llm_batch_size*n_gpu, 
                                        shuffle=False, num_workers=2*n_gpu, 
                                        worker_init_fn=init_tokenizer)
    for j, batch in enumerate(tqdm(train_loader, desc='Generating embeddings')):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                batch,
                max_length = args.model_max_length,
                return_tensors = 'pt',
                padding = 'longest',
                truncation = True #model will cut off sequences that are too long (>model_max_length tokens (default 400)
            )
            input_ids = token_feat['input_ids'].to(device)
            attention_mask = token_feat['attention_mask'].to(device) #which tokens are padding and which are important
            model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1) #gets mean pooling embeddings across all token embeddings of each sequence


            if j==0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())

    #reorder back to original order sequences were presented in
    embeddings = embeddings[np.argsort(idx)]
    #embeddings = embeddings.numpy()
    embeddings = normalize(embeddings)

    #write to csv file
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [''] + [str(i) for i in range(len(embeddings[0]))]
        writer.writerow(header)
        for contig_id, embedding in tqdm(zip(contig_ids, embeddings), total=len(contig_ids), desc="Writing to CSV"):
            writer.writerow([contig_id] + embedding.tolist())