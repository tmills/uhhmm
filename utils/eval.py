import os
import sys
import subprocess
import re
import numpy as np
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# EVALB folder needs to be placed into this folder

last_sample_folder = sys.argv[1]
gold_trees_file = sys.argv[2]
evalb_params_file = './utils/EVALB/uhhmm.prm'
def bash_command(cmd):
    return subprocess.Popen(['/bin/bash', '-c', cmd])

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
output_last_samples = [x for x in output_files if re.match('last_sample[0-9]+\.(fromdeps\.)?linetrees', x)]

rule_f_names = []
model_f_names = []
head_model_names = []
deps_f_names = []

qs = []
for f in output_last_samples:
    f_name = os.path.join(last_sample_folder, f)
    rule_f_name = f_name.replace('linetrees', 'rules')
    p = bash_command(build_rules_command.format(f_name, modelblocks_path, rule_f_name))
    rule_f_names.append(rule_f_name)
    qs.append(p)
for p in qs:
    p.wait()
qs = []
for index, rule_f_name in enumerate(rule_f_names):
    model_f_name = rule_f_name.replace('rules', 'model')
    p = bash_command(build_model_command.format(rule_f_name, rule_f_name, model_f_name))
    model_f_names.append(model_f_name)
    qs.append(p)
for p in qs:
    p.wait()
qs = []
for index, model_f_name in enumerate(model_f_names):
    head_model_name = model_f_name.replace('model', 'head.model')
    p = bash_command(build_head_model_command.format(model_f_name, modelblocks_path, head_model_name))
    head_model_names.append(head_model_name)
    qs.append(p)
for p in qs:
    p.wait()

qs = []
for index, f_name in enumerate(output_last_samples):
    f_name = os.path.join(last_sample_folder, f_name)
    deps_f_name = f_name.replace('linetrees', 'fromdeps.linetrees')
    p = bash_command(convert_to_deps_command.format(f_name, modelblocks_path, head_model_names[index], modelblocks_path, deps_f_name))
    deps_f_names.append(deps_f_name)
    qs.append(p)
for p in qs:
    p.wait()

end_f_names = []
ps = []
p2s = []
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
    ps.append(p)
for p in ps:
    p.wait()

evalb_f_names = []
print('running EVALB on the line trees')
for end_f_name in end_f_names:
    evalb_name = end_f_name.replace('linetrees', 'evalb')
    # print(eval_command.format(evalb_params_file, gold_trees_file, end_f_name, evalb_name))
    p2 = bash_command(eval_command.format(evalb_params_file, gold_trees_file, end_f_name, evalb_name))
    p2s.append(p2)
    evalb_f_names.append(evalb_name)
for p2 in p2s:
    p2.wait()

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
pp = PdfPages(last_sample_folder + 'evalb' + '_'+ str(min(indices))+'_' + str(max(indices)) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()

bash_command('rm -f {}'.format(' '.join(rule_f_names + model_f_names + head_model_names + deps_f_names + end_f_names + evalb_f_names)))

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