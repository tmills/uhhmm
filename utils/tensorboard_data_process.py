from tensorboardX import SummaryWriter
import argparse
import pandas
import re
import os
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-dir', required=True)
args = parser.parse_args()

output_dir = args.output_dir

writer = SummaryWriter()
print(writer.file_writer.get_logdir())
terms = pandas.read_table(output_dir + '/pcfg_terms.txt', '\t')
nonterms = pandas.read_table(output_dir + '/pcfg_nonterms.txt', '\t')
for col_name, column in nonterms.iteritems():
    # print(col_name)
    # print(column[0])
    if col_name == 'iter' or column.iloc[-1] < 0.05:
        continue
    else:
        p, c = col_name.split('->')
        c = c.replace('(', '')
        c = c.replace(')', '')
        c = c.replace(',', 'R')
        c = c.replace(' ', '')
        c = 'L' + c
        print(p, c)
        name = '/'.join(['nonterms', p, c])
        for val_ind, val in enumerate(column):
            # print(val, type(val), val_ind, 'nonterms/'+col_name)
            writer.add_scalar(name, float(val), val_ind)

for col_name, column in terms.iteritems():
    # print(col_name)
    # print(column[0])
    if col_name == 'iter' or column.iloc[-1] < 0.001:
        continue
    else:
        p, c = col_name.split('->')
        c = c.replace('(', '')
        c = c.replace(')', '')
        # c = c.replace(',', 'R')
        # c = c.replace(' ', '')
        # c = 'L' + c
        print(p, c)
        name = '/'.join(['terms', p, c])
        for val_ind, val in enumerate(column):
            # print(val, type(val), val_ind, 'nonterms/'+col_name)
            writer.add_scalar(name, float(val), val_ind)
writer.close()