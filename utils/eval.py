import os
import sys
import subprocess
from multiprocessing import Pool
import re
import numpy as np
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas
import time
import nltk
from collections import Counter
# EVALB folder needs to be placed into this folder

last_sample_folder = sys.argv[1]
gold_trees_file = sys.argv[2]
evalb_params_file = './utils/EVALB/uhhmm.prm'
total_process_limit = 20
process_pool = []
burn_in = 50

def bash_command(cmd):
    global process_pool
    # if len(process_pool) > 0:
    #     print(sum([(not bool(x.poll())) for x in process_pool]), process_pool[0].poll())
    while True:
        process_pool = [x for x in process_pool if x.poll() is None]
        if len(process_pool) >= total_process_limit:
            time.sleep(0.1)
            continue
        else:
            p = subprocess.Popen(['/bin/bash', '-c', cmd])
            process_pool.append(p)
            break
    return p

gold_trees = [nltk.Tree.fromstring(x.strip()) for x in open(gold_trees_file)]

modelblocks_path = open('user-modelblocks-location.txt').readline().strip()

build_rules_command = '''cat {}  |  perl {}/resource-linetrees/scripts/trees2rules.pl  >  {}'''
build_model_command = '''cat {0} | sort | uniq -c | sort -nr | awk '{{"wc -l {1} | cut -d \\" \\" -f1" | getline t; u = $1; $1 = u/t; print;}}' | awk '{{p = $1; for (i=1;i<NF;i++) $i=$(i+1);$NF="="; $(NF + 1)=p; tmp=$2;$2=$3;$3=tmp;$1="R";print;}}' > {2}'''
build_head_model_command =  '''cat {} | python3 {}/resource-linetrees/scripts/rules2headmodel.py > {}'''
convert_to_deps_command = '''cat {} | python3 {}/resource-linetrees/scripts/trees2deps.py {} | sed 's/<num>/-NUM-/g' | python {}/resource-linetrees/scripts/deps2trees.py -f stanford > {}'''

preprocess_command = '''cat {} | sed 's/(-DFL- \+E_S) *//g;s/  \+/ /g;s/\\t/ /g;s/\([^ ]\)(/\\1 (/g;s/_//g;s/-UNDERSCORE-//g;s/([^ ()]\+ \+\*[^ ()]*)//g;s/( *-NONE-[^ ()]\+ *[^ ()]* *)//g;s/([^ ()]\+ )//g;s/ )/)/g;s/( /(/g;s/  \+/ /g;' | awk '!/^\s*\(CODE/' | python {}/resource-linetrees/scripts/make-trees-lower.py | perl {}/resource-linetrees/scripts/killUnaries.pl | perl {}/resource-linetrees/scripts/make-trees-nounary.pl |  perl -pe "s/ \([^ ()]+ (,|\.|\`\`|\`|--|-RRB-|-LRB-|-LCB-|-RCB-|''|'|\.\.\.|\?|\!|\:|\;)\)//g" | perl {}/resource-linetrees/scripts/make-trees-nounary.pl > {}'''
# eval_command = '''python {}/resource-linetrees/scripts/constit_eval.py {} <(cat {} | python {}/resource-linetrees/scripts/filter_reannot_fails.py {}) > {}.nt-lower-nounary-nopunc.constiteval.txt'''
eval_command = './utils/EVALB/evalb -p {} {} {} > {}' # param, gold, test
constlist_command = '''python scripts/iters2constitevallist.py {} > {}'''
plot_command = '''python {}/resource-linetrees/scripts/constitevals2table.py {} > {} '''

output_files = os.listdir(last_sample_folder)
output_last_samples = [x for x in output_files if re.match('last_sample[0-9]+\.linetrees', x)]

output_last_samples.sort(key=lambda x: int(re.match('last_sample([0-9]+)\.linetrees', x).group(1)))
print(output_last_samples)

rule_f_names = []
model_f_names = []
head_model_names = []
deps_f_names = []

for f in output_last_samples:
    f_name = os.path.join(last_sample_folder, f)
    rule_f_name = f_name.replace('linetrees', 'rules')
    p = bash_command(build_rules_command.format(f_name, modelblocks_path, rule_f_name))
    rule_f_names.append(rule_f_name)

for index, rule_f_name in enumerate(rule_f_names):
    model_f_name = rule_f_name.replace('rules', 'model')
    p = bash_command(build_model_command.format(rule_f_name, rule_f_name, model_f_name))
    model_f_names.append(model_f_name)

for index, model_f_name in enumerate(model_f_names):
    head_model_name = model_f_name.replace('model', 'head.model')
    p = bash_command(build_head_model_command.format(model_f_name, modelblocks_path, head_model_name))
    head_model_names.append(head_model_name)


for index, f_name in enumerate(output_last_samples):
    f_name = os.path.join(last_sample_folder, f_name)
    deps_f_name = f_name.replace('linetrees', 'fromdeps.linetrees')
    p = bash_command(convert_to_deps_command.format(f_name, modelblocks_path, head_model_names[index], modelblocks_path, deps_f_name))
    deps_f_names.append(deps_f_name)


end_f_names = []

print('preprocessing the trees into nounary lower and nopunc')
for f in output_last_samples + deps_f_names:
    if 'fromdeps' not in f:
        f_name = os.path.join(last_sample_folder, f)
    else:
        f_name = f
    end_f_name = '{}.nt.lower.nounary.nopunc.linetrees'.format(f_name.replace('.linetrees', ''))
    end_f_names.append(end_f_name)
    # print(preprocess_command.format(f_name, modelblocks_path, modelblocks_path, modelblocks_path, modelblocks_path, end_f_name))
    p = bash_command(preprocess_command.format(f_name, modelblocks_path, modelblocks_path, modelblocks_path, modelblocks_path, end_f_name))


evalb_f_names = []
print('running EVALB on the line trees')
for end_f_name in end_f_names:
    evalb_name = end_f_name.replace('linetrees', 'evalb')
    # print(eval_command.format(evalb_params_file, gold_trees_file, end_f_name, evalb_name))
    p2 = bash_command(eval_command.format(evalb_params_file, gold_trees_file, end_f_name, evalb_name))
    evalb_f_names.append(evalb_name)


indices = []
prec = []
rec = []
f1 = []
deps_indices = []
deps_prec = []
deps_rec =[]
deps_f1 = []
print('plotting the p, r and f')
for evalb_name in evalb_f_names:
    index = re.search('(?<=last_sample)[0-9]+', evalb_name).group(0)
    if 'fromdeps' in evalb_name:
        deps_indices.append(int(index))
    else:
        indices.append(int(index))
    with open(evalb_name) as e:
        lines = e.readlines()
        for line in lines:
            if line.startswith('Bracketing Precision'):
                this_prec = re.search('[0-9\.]+', line).group(0)
                if 'fromdeps' in evalb_name:
                    deps_prec.append(float(this_prec))
                else:
                    prec.append(float(this_prec))
            if line.startswith('Bracketing Recall'):
                this_rec = re.search('[0-9\.]+', line).group(0)
                if 'fromdeps' in evalb_name:
                    deps_rec.append(float(this_rec))
                else:
                    rec.append(float(this_rec))
            if line.startswith('Bracketing FMeasure'):
                this_fm = re.search('[0-9\.]+', line).group(0)
                if 'fromdeps' in evalb_name:
                    deps_f1.append(float(this_fm))
                else:
                    f1.append(float(this_fm))
                break

titles = ['Precision', 'Recall', 'F1', 'Precision-deps', 'Recall-deps', 'F1-deps']
data = list(zip(indices, prec, rec, f1))
data.sort(key=operator.itemgetter(0))
data = np.array(data)

data_deps = list(zip(deps_indices, deps_prec, deps_rec, deps_f1))
data_deps.sort(key=operator.itemgetter(0))
data_deps = np.array(data_deps)

data = np.hstack((data, data_deps[:, 1:]))
fig, ax = plt.subplots()
lines = ax.plot(data[:, 0], data[:,1], data[:,0], data[:,2], data[:,0], data[:,3], data[:,0], data[:,4], data[:,0], data[:,5], data[:,0], data[:,6])
ax.set_ylabel('percentage')
ax.legend(lines, ('Precision', "Recall", "F1", 'Precision-deps', 'Recall-deps', 'F1-deps'))
pp = PdfPages(os.path.join(last_sample_folder, 'evalb' + '_'+ str(min(indices))+'_' + str(max(indices))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()


# plot the log probs
hyperparams = pandas.read_table(os.path.join(last_sample_folder, 'pcfg_hypparams.txt'))

fig, ax = plt.subplots()
line_1 = ax.plot(hyperparams.iter[burn_in:], hyperparams.logprob[burn_in:], 'b-')
ax2 = ax.twinx()
line_2 = ax2.plot(hyperparams.iter[burn_in:], np.nan_to_num(hyperparams.val_logprob[burn_in:]), 'r-')
ax.set_ylabel('logprob', color='b')
ax.tick_params('y', color='b')
ax2.set_ylabel('dev logprob', color='r')
ax2.tick_params('y',color='r')
ax.set_xlabel('iteration')
# ax.legend(lines, ('Training', 'Dev'))
fig.tight_layout()
pp = PdfPages(os.path.join(last_sample_folder, 'logprobs' + '_'+ str(min(hyperparams.iter[burn_in:]))+'_' + str(max(hyperparams.iter[burn_in:]))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()


# NP, VP and PP evaluations
phrases = ['NP', 'VP', 'PP']
gold_counters = {'NP':Counter(), 'VP':Counter(), 'PP':Counter()}
for t in gold_trees:
    for sub_t in t.subtrees():
        if sub_t.label() in phrases and len(sub_t.leaves()) > 1:
            gold_counters[sub_t.label()][' '.join(sub_t.leaves())] += 1

scores = []
aggregate_scores = []
NP_length = sum(gold_counters['NP'].values())
VP_length = sum(gold_counters['VP'].values())
PP_length = sum(gold_counters['PP'].values())
phrases_lengths = [NP_length, VP_length, PP_length]

def calc_phrase_stats(f_name, prec_thres=0.6):
    current_counters = {}
    with open(f_name) as fh:
        for line in fh:
            this_t = nltk.Tree.fromstring(line.strip())
            for sub_t in this_t.subtrees():
                if  len(sub_t.leaves()) > 1:
                    if sub_t.label() not in current_counters:
                        current_counters[sub_t.label()] = Counter()
                    current_counters[sub_t.label()][' '.join(sub_t.leaves())] += 1
    this_best_scores = [0,0,0]
    this_best_aggregate_scores = [0, 0, 0]
    best_cats = [0,0,0]
    aggregate_counters = [Counter(), Counter(), Counter()]
    for cat, cat_counter in current_counters.items():
        for index, phrase_cat in enumerate(phrases):
            acc_num = sum(( cat_counter & gold_counters[phrase_cat]).values())
            prec = acc_num / sum(cat_counter.values())
            rec = acc_num / phrases_lengths[index]
            f1 = 0 if prec == 0 or rec == 0 else 1.0 / (0.5 / prec + (0.5 / rec))
            if f1 > this_best_scores[index]:
                this_best_scores[index] = f1
                best_cats[index] = cat
            if prec > prec_thres:
                aggregate_counters[index] += cat_counter
    for i in range(len(aggregate_counters)):
        aggregate_counters[i] += current_counters[best_cats[i]]
    for index, phrase_cat in enumerate(phrases):
        acc_num = sum((aggregate_counters[index] & gold_counters[phrase_cat]).values())
        prec = acc_num / sum(aggregate_counters[index].values())
        rec = acc_num / phrases_lengths[index]
        f1 = 0 if prec == 0 or rec == 0 else 1.0 / (0.5 / prec + (0.5 / rec))
        this_best_aggregate_scores[index] = f1
    return this_best_scores, this_best_aggregate_scores

with Pool(processes=total_process_limit) as pool:
    scores, aggregate_scores = zip(*(pool.map(calc_phrase_stats, [os.path.join(last_sample_folder, f) for f in output_last_samples])))

print(scores[0], aggregate_scores[0])
scores = np.array(scores)
aggregate_scores = np.array(aggregate_scores)
fig, ax = plt.subplots()
len_iters = len(output_last_samples)
x_data = hyperparams.iter[:len_iters]
lines = ax.plot(x_data, scores[:len_iters, 0], x_data, scores[:len_iters, 1], x_data, scores[:len_iters, 2]
                ,x_data, aggregate_scores[:len_iters, 0] , x_data, aggregate_scores[:len_iters, 1],
                x_data, aggregate_scores[:len_iters, 2])
ax.set_ylabel('percentage')
ax.legend(lines, ( "best NP F1", 'best VP F1', 'best PP F1', "best NP agg F1", "best VP agg F1", "best PP agg F1"))
pp = PdfPages(
    os.path.join(last_sample_folder, 'phrases' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()

with Pool(processes=total_process_limit) as pool:
    scores, aggregate_scores = zip(*(pool.map(calc_phrase_stats, deps_f_names)))

print(scores[0], aggregate_scores[0])
scores = np.array(scores)
aggregate_scores = np.array(aggregate_scores)
fig, ax = plt.subplots()
lines = ax.plot(x_data, scores[:len_iters, 0], x_data, scores[:len_iters, 1], x_data, scores[:len_iters, 2]
                ,x_data, aggregate_scores[:len_iters, 0] , x_data, aggregate_scores[:len_iters, 1],
                x_data, aggregate_scores[:len_iters, 2])
ax.set_ylabel('percentage')
ax.legend(lines, ( "best NP F1 deps", 'best VP F1 deps', 'best PP F1 deps', "best NP agg F1 deps", "best VP agg F1 deps", "best PP agg F1 deps"))
pp = PdfPages(
    os.path.join(last_sample_folder, 'phrases_deps' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()

# delete the temp files
for f in rule_f_names + model_f_names + head_model_names + deps_f_names + end_f_names + evalb_f_names:
    bash_command('rm -f {}'.format(f))

# output_files = os.listdir(last_sample_folder)
# output_last_samples = [x for x in output_files if x.endswith('.constiteval.txt')]
# f_list = []
# for f in output_last_samples:
#     f_list.append(os.path.join(last_sample_folder, f))
# f_list_str = ' '.join(f_list)
# f_constlist = os.path.join(last_sample_folder, 'nt-lower-nounary-nopunc.uhhmm-iter.constitevallist')
# bash_command(constlist_command.format(f_list_str, f_constlist))
#
# f_const_table = os.path.join(last_sample_folder, f_constlist, 'nt-lower-nounary-nopunc.uhhmm-iter.constitevaltable.txt')
# bash_command(plot_command.format(modelblocks_path, f_const_table))