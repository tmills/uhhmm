from tensorboardX import SummaryWriter
import argparse
import pandas
import re
import os
import sys
sys.path.append('./scripts/')
from pcfg_translator import load_gold_trees

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-dir', required=True)
parser.add_argument('-nd', '--domain-size', default=0, type=int)
args = parser.parse_args()

output_dir = args.output_dir

writer = SummaryWriter()
print(writer.file_writer.get_logdir())
terms = pandas.read_table(output_dir + '/pcfg_terms.txt', '\t')
nonterms = pandas.read_table(output_dir + '/pcfg_nonterms.txt', '\t')

if args.domain_size:
    for last_sample_fn in os.listdir(output_dir):
        if 'last_sample' in last_sample_fn:
            number = int(re.search('[0-9]+', last_sample_fn).group(0))
            _, counts = load_gold_trees(os.path.join(output_dir, last_sample_fn), args.domain_size)
            for lhs in counts:
                parent = lhs.symbol()
                for rhs in counts[lhs]:
                    if len(rhs) > 1:
                        children = 'L'+ rhs[0].symbol() + 'R' + rhs[1].symbol()
                    else:
                        children = rhs[0]
                    if len(rhs) > 1:
                        name = '/'.join(['nontermsC', parent, children])
                        writer.add_scalar(name, counts[lhs][rhs], number)
                    else:
                        name = '/'.join(['termsC', parent, children])
                        writer.add_scalar(name, counts[lhs][rhs], number)

for col_name, column in nonterms.iteritems():
    # print(col_name)
    # print(column[0])
    if col_name == 'iter':# or column.iloc[-1] < 0.05:
        continue
    else:
        p, c = col_name.split('->')
        c = c.replace('(', '')
        c = c.replace(')', '')
        c = c.replace(',', 'R')
        c = c.replace(' ', '')
        c = 'L' + c
        # print(p, c)
        name = '/'.join(['nonterms', p, c])
        for val_ind, val in enumerate(column):
            # print(val, type(val), val_ind, 'nonterms/'+col_name)
            writer.add_scalar(name, float(val), val_ind)

for col_name, column in terms.iteritems():
    # print(col_name)
    # print(column[0])
    if col_name == 'iter':# or column.iloc[-1] < 0.001:
        continue
    else:
        p, c = col_name.split('->')
        c = c.replace('(', '')
        c = c.replace(')', '')
        # c = c.replace(',', 'R')
        # c = c.replace(' ', '')
        # c = 'L' + c
        # print(p, c)
        name = '/'.join(['terms', p, c])
        for val_ind, val in enumerate(column):
            # print(val, type(val), val_ind, 'nonterms/'+col_name)
            writer.add_scalar(name, float(val), val_ind)
writer.close()