# this file is for add nonterminals 'X' to output trees from systems that do
# not have any labels, for example CCL and UPPARSE

import sys
import re
fn = sys.argv[1]

with open(fn) as i, open(fn+'.proper', 'w') as o:
    for line in i:
        line = re.sub('\(\s*', '(X ', line)
        print(line, file=o)