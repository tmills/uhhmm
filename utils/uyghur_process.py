from nltk.probability import FreqDist
from scipy.stats import describe
import os
import xml.etree.ElementTree as ET
'''
this file is used for filtering some data set (in this case the uyghur set of DARPA)
given some frequent words and sentence length limit, effectively shrinking the size of the
dataset.
'''

first_n_words = None  # first N words in terms of frequency. set None to include all words
thres = 1000    # the length limit of a sentence

set0_file_dir = '/home/jin.544/project_space/jin/set0/data/translation/found/il3/ltf/'
ltf_files = [ os.path.join(set0_file_dir, x) for x in os.listdir(set0_file_dir) if 'ltf.xml' in x ]
word_dist = FreqDist()
sent_tokens = []
for f_name in ltf_files:
    xml_tree = ET.parse(f_name)
    root =xml_tree.getroot()
    for seg in root.iter(tag='SEG'):
        this_seg = []
        for token in seg.iter(tag='TOKEN'):
            if first_n_words is not None:
                word_dist[token.text] += 1
            this_seg.append(token.text)
        sent_tokens.append(this_seg)

if first_n_words is not None:
    good_vocab = word_dist.most_common(first_n_words)
    good_vocab = set(x[0] for x in good_vocab)
else:
    good_vocab = set()

result_file = '/home/jin.544/project_space/jin/set0/data//translation/found/il3/rsd/uyghur.filtered.txt'
k = 0
with open(result_file, 'w', encoding='utf8') as w:
    for line in sent_tokens:
        if len(line) > thres or not any(line):# or len(line) == 1:
            continue
        if good_vocab:
            for word in line:
                if word not in good_vocab:
                    break
            else:
                k += 1
                print(' '.join(line), file=w)
        else:
            k += 1
            print(' '.join(line), file=w)
print('total {} lines in the final filtered file'.format(k))