from nltk.probability import FreqDist
from scipy.stats import describe

'''
this file is used for filtering some data set (in this case the uyghur set of DARPA)
given some frequent words and sentence length limit, effectively shrinking the size of the
dataset.
'''
set0_file = '/home/jin.544/project_space/jin/set0/data/monolingual_text/ltf/uyghur_set0_all.rsd.txt'
word_dist = FreqDist()
f = open(set0_file, encoding='utf8')
lengths = []
for line in f:
    line = line.strip().split(' ')
    lengths.append(len(line))
    for word in line:
        word_dist[word] += 1
print(describe(lengths))
print(word_dist.B())
f.seek(0)

good_vocab = word_dist.most_common(60000)
good_vocab = set(x[0] for x in good_vocab)

thres = 20
result_file = '/home/jin.544/project_space/jin/set0/data/monolingual_text/ltf/uyghur_set0_filtered.rsd.txt'
k = 0
with open(result_file, 'w', encoding='utf8') as w:
    for line in f:
        line = line.strip().split(' ')
        if len(line) > thres or not any(line) or len(line) == 1:
            continue
        for word in line:
            if word not in good_vocab:
                break
        else:
            k += 1
            print(' '.join(line), file=w)
print('total {} lines in the final filtered file'.format(k))