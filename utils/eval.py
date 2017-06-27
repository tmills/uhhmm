import os
import sys
import subprocess

last_sample_folder = sys.argv[1]
gold_trees_file = sys.argv[2]

modelblocks_path = open('user-modelblocks-location.txt').readline().strip()

preprocess_command = '''cat {} | sed 's/(-DFL- \+E_S) *//g;s/  \+/ /g;s/\t/ /g;s/\([^ ]\)(/\1 (/g;s/_//g;s/-UNDERSCORE-//g;s/([^ ()]\+ \+\*[^ ()]*)//g;s/( *-NONE-[^ ()]\+ *[^ ()]* *)//g;s/([^ ()]\+ )//g;s/ )/)/g;s/( /(/g;s/  \+/ /g;' | awk '!/^\s*\(CODE/' | python /home/scratch/jin/modelblocks-release/resource-linetrees/scripts/make-trees-lower.py | perl /home/scratch/jin/modelblocks-release/resource-linetrees/scripts/killUnaries.pl | perl /home/scratch/jin/modelblocks-release/resource-linetrees/scripts/make-trees-nounary.pl |  perl -pe "s/ \([^ ()]+ (,|\.|\`\`|\`|--|-RRB-|-LRB-|-LCB-|-RCB-|''|'|\.\.\.|\?|\!|\:|\;)\)//g" | perl /home/scratch/jin/modelblocks-release/resource-linetrees/scripts/make-trees-nounary.pl > {}.nt.lower.nounary.nopunc.linetrees'''
eval_commnad = '''python {}/resource-linetrees/scripts/constit_eval.py {} <(cat {} | python {}/resource-linetrees/scripts/filter_reannot_fails.py {}) > {}.nt-lower-nounary-nopunc.constiteval.txt'''

constlist_command = '''python /home/mulproj/jin/uhhmm//scripts/iters2constitevallist.py {} > {}'''
plot_command = '''python {}/resource-linetrees/scripts/constitevals2table.py {} > {} '''

output_files = os.listdir(last_sample_folder)
output_last_samples = [x for x in output_files if x.endswith('linetrees')]
for f in output_last_samples:
    f_name = os.path.join(last_sample_folder, f)
    end_f_name = '{}.nt.lower.nounary.nopunc.linetrees'.format(f_name.replace('.linetrees', ''))
    subprocess.call(preprocess_command.format(f_name, end_f_name))
    subprocess.call(eval_commnad.format(modelblocks_path, gold_trees_file, end_f_name, modelblocks_path, gold_trees_file, f_name.replace('.linetrees', '')))

output_files = os.listdir(last_sample_folder)
output_last_samples = [x for x in output_files if x.endswith('.constiteval.txt')]
f_list = []
for f in output_last_samples:
    f_list.append(os.path.join(last_sample_folder, f))
f_list_str = ' '.join(f_list)
f_constlist = os.path.join(last_sample_folder, 'nt-lower-nounary-nopunc.uhhmm-iter.constitevallist')
subprocess.call(constlist_command.format(f_list_str, f_constlist))

f_const_table = os.path.join(last_sample_folder, f_constlist, 'nt-lower-nounary-nopunc.uhhmm-iter.constitevaltable.txt')
subprocess.call(plot_command.format(modelblocks_path, f_const_table))