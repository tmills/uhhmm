#!/usr/bin/env python3

import seaborn as sea
import re, itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from left_corner2normal_tree_converter import parse_state_seq_string, normalize_dateline
import sys

"""
this is the diagnostics generation script which plots trace plots for all models, logprobs,
left and right branching tendencies in sampled trees, depth usage and so on
plots are generated with Seaborn and saved as pdf
"""

def parse_model_file(model_file, max_cat):
    model = {}
    conds = set()
    rvs = set()
    max_invalid_cat = str(max_cat + 1)
    with open(model_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            else:
                expression, value = line.split(' = ')
                value = float(value)
                items = re.search('P\( ([^\s]+\d*) \| (\w*\(.+\)), (-?\d) \)', expression)
                dependent = items.group(1)
                if (all([x not in model_file for x in ['lex', 'p_j', 'fork']]) and max_invalid_cat not in dependent) or \
                    any([x in model_file for x in ['lex', 'p_j', 'fork']]):
                    rvs.add(dependent)
                    independent = items.group(2)
                    if max_invalid_cat not in independent:
                        conds.add(independent)
                    depth = items.group(3)
                    model[(dependent, independent, int(depth))] = value
    return model, conds, rvs

def build_timeseries_data(models, conds, rvs, max_depth=0):
    timesteps = len(models)
    records = []
    if max_depth <= 0:
        column_names = ('v|c', 'timestep', 'probs')
    else:
        column_names = ('v|c', 'timestep', 'probs', 'depth')
    for model_index, model in enumerate(models):
        for cond_index,cond in enumerate(conds):
            for rv_index, rv in enumerate(rvs):
                if max_depth <=0:
                    if (rv, cond, max_depth) in model:
                        records.append((rv+'|'+cond, model_index, model[(rv, cond, max_depth)]))
                    else:
                        records.append((rv + '|' + cond, model_index, 0))
                else:
                    for depth in range(0, max_depth):
                        if (rv, cond, depth) in model:
                            records.append((rv + '|' + cond, model_index, model[(rv, cond, depth)], depth))
                        else:
                            records.append((rv + '|' + cond, model_index, 0, depth))
    timeseries = pd.DataFrame.from_records(records, columns=column_names)
    return timeseries

def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=True, **kwargs)

def plot_trace(timeseries, num_rvs):
    sea.set(style="darkgrid")
    g = sea.FacetGrid(timeseries, col="v|c", col_wrap=num_rvs, size=2)
    g = g.map_dataframe(dateplot, "timestep", "probs", linewidth=0.5)
    return g

def list_files(dir, suffix, maximum=5000, burn_in = 50):
    file_list = []
    for i in range(burn_in, maximum):
        if os.path.exists(dir+suffix+str(i)+'.txt'):
            file_list.append(dir+suffix+str(i)+'.txt')
        else:
            pass
    return file_list

def plot_trace_main(suffix, depth, dir, model_name, max_cat = 10):
    file_list = list_files(dir, suffix)
    conds = set()
    rvs = set()
    models = []
    for file in file_list:
        this_model, this_conds, this_rvs = parse_model_file(file, max_cat)
        models.append(this_model)
        conds.update(this_conds)
        rvs.update(this_rvs)
    conds = sorted(list(conds))
    rvs = sorted(list(rvs))
    timeseries = build_timeseries_data(models, conds, rvs, depth)
    num_rvs = len(rvs)
    if num_rvs == 2:
        num_rvs = num_rvs * 3
    elif num_rvs > 6:
        num_rvs = 6
    if depth <=0 :
        plot = plot_trace(timeseries, num_rvs)
        pp = PdfPages(dir+model_name+".pdf")
        sea.plt.savefig(pp, format='pdf')
        pp.close()
    else:
        for this_depth in range(0, depth):
            plot = plot_trace(timeseries[timeseries.depth== this_depth], num_rvs)
            pp = PdfPages(dir+model_name+str(this_depth)+".pdf")
            sea.plt.savefig(pp, format='pdf')
            pp.close()

def plot_log_probs(dir):
    logprobs = pd.read_table(dir+'logprobs.txt', header=0, index_col=0 )
    plt.plot(logprobs)
    pp = PdfPages(dir + "logprobs.pdf")
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.clf()

def plot_depth2_branching(dir, depth=2):
    file_list = list_files(dir, 'last_sample')
    d2 = []
    lr_odds = []
    for file in file_list:
        with open(file) as f:
            total = 0
            num_d2_percent = 0
            lb_rb_odds = [0, 0]
            for line in f:
                line = line.strip()
                states, words = parse_state_seq_string(line, depth)
                states = normalize_dateline(states)
                for index, state in enumerate(states):
                    if state.f == 1 and state.j == 1:
                        lb_rb_odds[1] += 1
                    elif state.f == 0 and state.j == 0:
                        lb_rb_odds[0] += 1
                    if index != 0 and state.f == 1 and state.j == 0:
                        num_d2_percent += 1
                    total += 1
            num_d2_percent = num_d2_percent / total
            lb_rb_odds = lb_rb_odds[0] / (lb_rb_odds[0] + lb_rb_odds[1]) # left_branching / total
        d2.append(num_d2_percent)
        lr_odds.append(lb_rb_odds)
    pp = PdfPages(dir + 'd2_lr.pdf')
    plt.subplot(121)
    plt.plot(d2)
    plt.xlabel('iters')
    plt.ylabel('proportion of d2 sents')
    plt.title('d2 num')
    plt.subplot(122)
    plt.plot(lr_odds)
    plt.xlabel('iters')
    plt.ylabel('left/total branching ratio')
    plt.title('l/all ratio')
    plt.savefig(pp, format='pdf')
    pp.close()

if __name__ == '__main__':
    suffixes = ['p_fork_','p_j_', 'p_act_', 'p_awa_j0_', 'p_awa_j1_', 'p_pos_']
    # suffixes = ['p_lex_given_pos'] # too many dependent vars. too slow
    dir = sys.argv[1]
    depth = int(sys.argv[2])
    abp_domain_size = int(sys.argv[3])
    plot_log_probs(dir)
    plot_depth2_branching(dir)
    depth = [depth,depth,depth,depth,depth,-1,-1]
    for index, suf in enumerate(suffixes):
        plot_trace_main(suf, depth[index], dir, suf, abp_domain_size)
