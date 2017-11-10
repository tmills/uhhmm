# this file is for add nonterminals 'X' to output trees from systems that do
# not have any labels, for example CCL and UPPARSE

import sys
import re
fn = sys.argv[1]

with open(fn) as i, open(fn+'.proper', 'w') as o:
    for line in i:
        all_words = re.findall('[ \(]{1}[^\(\)\s]+[\) ]{1}', line)
        for word in all_words:
            if word.startswith('(') and word.endswith(')'):
                continue
            else:
                line = line.replace(word, ' (' + word.strip() + ') ')
        line = re.sub('\(\s*', '(X ', line)
        o.write(line)