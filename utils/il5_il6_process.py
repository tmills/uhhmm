import sys
filename = sys.argv[1]

with open(filename, encoding='utf8') as i, open(filename+'.clean.txt', 'w', encoding='utf8') as o:
    for line in i:
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