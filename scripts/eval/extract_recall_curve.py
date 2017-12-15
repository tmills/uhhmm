import sys, re, argparse

argparser = argparse.ArgumentParser('''
Extracts recall values from a user-supplied list of synevals and outputs a space-delimited table of recall measures.
''')
argparser.add_argument('synevals', type=str, nargs='+', help='One or more *.syneval files from which to extract recall measures')
argparser.add_argument('-m', '--measure', dest='measure', action='store', choices=['recall','fscore','precision'], default='recall', help='Accuracy measure to extract.')
args, unknown = argparser.parse_known_args()

iternum = re.compile('([0-9]+)-(uhhmm|dmv)')
measure = re.compile('= *([0-9]+\.[0-9]+)')

print('iter ' + args.measure)

for path in args.synevals:
    with open(path, 'rb') as f:
        iter = iternum.search(path).group(1)
        for line in f:
            if args.measure == 'recall':
                key = 'Bracketing Recall'
            elif args.measure == 'precision':
                key = 'Bracketing Precision'
            elif args.measure == 'fscore':
                key = 'Bracketing FMeasure'
            if line.startswith(key):
                x = measure.search(line).group(1)
                print (str(iter) + ' ' + str(x))
                break
