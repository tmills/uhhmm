import sys

segmenter = '----------'
initial = '0 0;BOG.0/BOG.0;-.0/-.0;-.0/-.0;-.0/-.0;-.0;0/0/0,0/0/0,0/0/0,0/0/0'
#sents =["['-/- 0/0:9', '+/- 7/2:1', '+/+ 7/5:14', '+/+ 7/3:10', '+/+ 7/3:8', '+/+ 7/7:15', '+/+ 7/3:13', '+/+ 7/4:4']",] 
emptyCat = "-.0/-.0"
relations = '0/0/0,0/0/0,0/0/0,0/0/0'
#parseFile = 'Untitled.txt'
#outputFile = 'ou.txt'
#for line in sys.stdin:
#out = open(outputFile, 'w')
reference = open(sys.argv[1]).readlines()
for sentnum, line in enumerate(sys.stdin):
	string = []
	string.append(segmenter)
	string.append(initial)
	refSentWords = reference[sentnum].split(' ')
	line = line.strip()
	line = line[2:-2]
	line = line.split("', '")
	#print line
	stack = [emptyCat]*4
	#print stack
	for index, step in enumerate(line):
		thisStep = []
		step = step.split(' ')
		print step
		fj = step[0].split('/')
		print fj
		thisStep.append(str(index+1))
		if fj[0] == '+':
			thisStep.append(str(0))
		else:
			thisStep.append(str(1))
		cats = step[1].split(':')
		if cats[0] != '0/0':
			preterm = '/'.join([x+'_' for x in cats[0].split('/')])
			if fj[1] == '-' and fj[0] == '+':
				for depth, element in enumerate(stack):
					if '-' in element:
						stack[depth] = preterm
						break
			elif (fj[1] == '+' and fj[0] == '+') or (fj[1] == '-' and fj[0] == '-'):
				for depth, element in enumerate(stack):
					if '-' in element:
						stack[depth-1] = preterm
						break
		thisStep.append(';'.join(stack))
		thisStep.append(cats[1]+'_')
		thisStep.append(relations)
		thisStep.append(refSentWords[index])
		#print thisStep
		string.append(thisStep)
	else:
		string.append([str(int(string[-1][0])+1), '1', "-.0/-.0;-.0/-.0;-.0/-.0;-.0/-.0", '99_', relations, 'null'])
		string.append(segmenter)
	for subs in string[::-1]:
		if subs != segmenter and subs != initial:
			print subs[0], (';').join(subs[1:-1]), subs[-1]
		else:
			print subs
#out.close()