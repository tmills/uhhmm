import re
'''
this file is used for filtering some data set (in this case the uyghur set of DARPA)
given some frequent words and sentence length limit, effectively shrinking the size of the
dataset.
'''
input_file = 'chinese.txt'
result_file = 'chinese.filtered.txt'
k = 0
with open(result_file, 'w', encoding='utf8') as w, open(input_file, encoding='utf8') as i:
    for line in i:
        words = line.strip().split(' ')
        if len(words) > 40 or len(words) < 2 or all(['punct' in x for x in words]) or all([
            (not bool(re.search(u"[\u4e00-\u9fff]", x))) for x in words
        ]) or line.count('unknown') > 3:
            continue
        else:
            line = line.replace('(', '-LRB-')
            line = line.replace(')', '-RRB-')
        k += 1
        w.write(line)

print('total {} lines in the final filtered file'.format(k))