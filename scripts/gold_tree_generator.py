import sys

for line in sys.stdin:
    line = line.strip().split()
    if len(line) == 3:
        sys.stdout.write('(5 (4 (1 ' + line[0] + ') (2 ' + line[1] + ')) (3 ' + line[2] + ')))' + '\n')
    elif len(line) == 7:
        sys.stdout.write('(5 (4 (1 ' + line[0] + ') (2 ' + line[1] + ')) (3 (3 (3 ' + line[2] +') (4 (1 '+ line[3] +
                         ') (2 '+line[4] + '))) (4 (1' + line[5] + ') (2 ' + line[6] + '))))' + '\n')
