import os
import csv

def to_cami_format(out_dir, out_name):
    infile = os.path.join(out_dir, 'comebin_res/comebin_res.tsv')
    outfile = os.path.join(out_dir, f'comebin_res/{out_name}.binning')
    if os.path.exists(outfile):
        os.remove(outfile)
        print(f'{out_name} already exists, deleting it and starting from scratch')

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

        for row in reader:
            print(row)
            out_file.write(f'{row[0]}\t{row[1]}\n')

if __name__=='__main__':
    to_cami_format('/Users/jackvaska/Desktop/BMI/capstone_project/COMEBin/comebin_test_data/excepted_output/run_comebin_test', 'marmg_test12')


