
import sys

segmenter = '----------'
initial = '0 0;BOG.0/BOG.0;-.0/-.0;-.0/-.0;-.0/-.0;-.0;0/0/0,0/0/0,0/0/0,0/0/0'
#sents =["['-/- 0/0:9', '+/- 7/2:1', '+/+ 7/5:14', '+/+ 7/3:10', '+/+ 7/3:8', '+/+ 7/7:15', '+/+ 7/3:13', '+/+ 7/4:4']",] 
emptyCat = "-.0/-.0"
relations = '0/0/0,0/0/0,0/0/0,0/0/0'

for sentnum, line in enumerate(sys.stdin):
	string = []
	string.append(segmenter)
	string.append(initial)

	line = line.strip()
	tokens = line.split(" ")

	stack = [emptyCat]*4
	for index, step in enumerate(tokens):
		print(index)
		thisStep = []
		step = step.split('::')
		fj = step[0].split('/')
		if index == 1:
			fj = ('+', '-')
		thisStep.append(str(index+1))
		if len(fj[0]) > 0 and fj[0] == '+':
			thisStep.append(str(0))
		else:
			thisStep.append(str(1))
		if step[1].endswith(':'):
			step[1] = step[1][:-1]+'COLON'
		elif step[1].endswith('('):
			step[1] = step[1][:-1]+'LB'
		elif step[1].endswith(')'):
			step[1] = step[1][:-1]+ 'RB'
		cats = step[1].split(':')
		if cats[0] != 'ACT0/AWA0':
			preterm = '/'.join([x+'_' for x in cats[0].split(';')[-1].split('/')])
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
			elif fj[1] == '+' and fj[0] == '-':
				for depth, element in enumerate(stack):
					if '-' in element:
						stack[depth-2] = preterm
						stack[depth-1] = emptyCat
						break
                    
		thisStep.append(';'.join(stack))
		tagword = cats[1].split(';')
		thisStep.append(tagword[0]+'_')
		thisStep.append(relations)
		if tagword[1] == 'COLON':
			tagword[1] = ':'
		thisStep.append(tagword[1] if len(tagword) > 1 else "Word%d"% index)
		#print thisStep
		string.append(thisStep)
	else:
		string.append([str(int(string[-1][0])+1), '1', "-.0/-.0;-.0/-.0;-.0/-.0;-.0/-.0", '99_', relations, 'null'])
#		string.append(segmenter)
	for subs in string[::-1]:
		if subs != segmenter and subs != initial:
			print subs[0], (';').join(subs[1:-1]), subs[-1]
		else:
			print subs
#out.close()
