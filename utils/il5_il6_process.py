import sys
# take the words file as input, not tagwords
filename = sys.argv[1]
indexfilename = sys.argv[2]

with open(filename, encoding='utf8') as i, open(filename+'.clean.txt', 'w', encoding='utf8') as o, open(indexfilename, encoding='utf8') as index_i,\
    open(indexfilename+'.clean.txt', 'w') as index_w:
    all_indices = index_i.readlines()
    for index, line in enumerate(i):
        segs = line.strip().split(' ')
        if len(segs) == 1 or len(segs) > 40:
            continue
        good_segs = []
        for seg in segs:
            if seg.startswith('#') or seg.startswith('http') or seg.startswith('@'):
                continue
            else:
                good_segs.append(seg.lower())
        print(' '.join(good_segs), file=o)
        print(all_indices[index].strip(), file=index_w)