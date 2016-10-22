import sys, re, argparse

argparser = argparse.ArgumentParser('''
Counts instances of a user-supplied depth level in each of a list of UHHMM parser last_sample*.txt input files.
''')
argparser.add_argument('samples', nargs='+', help='List of one or more UHHMM parser last_sample*.txt files')
argparser.add_argument('-d', '--depth', dest='d', action='store', default=2, help='Target depth level for counting (default=2)')
argparser.add_argument('-g', '--geq', dest='g', action='store_true', help='Count instances of depth greater than or equal to the target depth (defaults to counting depths exactly equal to the target)')
args, unkown = argparser.parse_known_args()

d = int(args.d)
get_iter = re.compile('last_sample([0-9]+).txt')

def add_depth_counts(string, depth_counts):
    depth = 0
    for i in range(len(string)):
        if string[i] == '+':
            if string[i+1:i+3] == '/-':
                depth += 1
                if len(depth_counts) <= depth:
                    depth_counts.append(1)
                else:
                    depth_counts[depth] += 1
        elif string[i] == '-':
            if string[i+1:i+3] == '/+':
                depth -= 1
    return depth_counts

count_field = 'depth_'
if args.g: count_field += 'geq'
count_field += str(d) + '_count'
            
print('iter ' + count_field)
            
for path in args.samples:
    iter = get_iter.search(path).group(1)
    with open(path, 'rb') as s:
        depth_counts = [0]
        line = s.readline()
        while line:
            depth_counts = add_depth_counts(line.strip(), depth_counts)
            line = s.readline()
    if d >= len(depth_counts):
        count = 0
    else:
        if args.g:
            count = sum(depth_counts[d:])
        else:
            count = depth_counts[d]
    print(iter + ' ' + str(count))
