#!/usr/bin/env python

import os
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig #For transformers v>4.28

def run_gen_dnabert(fasta_file, model_path):
    outfile = os.path.join(os.path.dirname(fasta_file), 'dnabert2_embeddings.csv')
    if model_path is None:
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)



    pass