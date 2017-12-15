import sys

for line in sys.stdin:
#	line = line.replace('$LTB$', '<')
#	line = line.replace('$RTB$', '>')
#	line = line.replace('$LSB$', '[')
#	line = line.replace('$RSB$', ']')
#	line = line.replace('$CARAT$', '^')
#	line = line.replace('$UNDERSCORE$', '_')
	sys.stdout.write(line)