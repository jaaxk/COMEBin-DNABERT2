import os
import csv
from tqdm import tqdm

def to_cami_format(out_dir, out_name):
    infile = os.path.join(out_dir, 'comebin_res/comebin_res.tsv')
    outfile = os.path.join(out_dir, f'cami_format/{out_name}.binning')
    if os.path.exists(outfile):
        os.remove(outfile)
        print(f'{out_name} already exists, deleting it and starting from scratch')

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    if 'marmg' in out_name:
        sampleid = 'marmgCAMI2_short_read_pooled_gold_standard_assembly'
    elif 'rhimg' in out_name:
        sampleid = 'rhimgCAMI2_short_read_pooled_gold_standard_assembly'
    elif 'strmg' in out_name:
        sampleid = 'strmgCAMI2_short_read_pooled_gold_standard_assembly'
    else:
        sampleid = out_name
        print('Leaving sampleID as: '+sampleid)

    header = '#CAMI Submission for Binning\n'
    header += f'@SampleID:{sampleid}\n'
    header += '@@SEQUENCEID\tBINID\n'

    with open(outfile, 'w') as out_file, open(infile, 'r') as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        out_file.write(header)

        for row in tqdm(reader):
            out_file.write(f'{row[0]}\t{row[1]}\n')

    print(f'.binning file generated at {outfile}')