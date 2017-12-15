#!/usr/bin/python
#this program will run through all of the .conllu files in each language folder and run them through conll2treebank.pl
import os
import sys

for f in os.listdir('.'):
	if (f[0] == 'U' and f[1] == 'D'):
#uncomment next line to delete all converted files first
#		os.system("rm " + f + "/convert*")
		for con in os.listdir(f):
			if (con[-1] == 'u' and con[-2] == 'l' and con[-3] == 'l'):
				os.system("./conll2treebank.pl < " + f + "/" + con +" > " + f + "/converted_" + con + ".txt" )
