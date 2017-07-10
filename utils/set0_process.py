from nltk.probability import FreqDist
set0_file = '/home/jin.544/project_space/jin/set0/data/monolingual_text/ltf/uyghur_set0_all.rsd.txt'
word_dist = FreqDist()
f = open(set0_file, encoding='utf8')
for line in f:
    line = line.strip().split(' ')
    for word in line:
        word_dist[word] += 1

f.seek(0)

good_vocab = word_dist.r_Nr(bins=10)
thres = 20
result_file = '/home/jin.544/project_space/jin/set0/data/monolingual_text/ltf/uyghur_set0_filtered.rsd.txt'

with open(result_file, 'w', encoding='utf8') as w:
    for line in f:
        line = line.strip().split(' ')
        if len(line) > thres:
            continue
        for word in line:
            if word not in good_vocab:
                break
        else:
            print(' '.join(line), file=w)
