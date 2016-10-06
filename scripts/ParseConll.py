#!/usr/bin/env python3

import pickle
import HmmParser
import sys
import logging
import os.path
from State import State

def main(args):
    if len(args) < 3:
        sys.stderr.write("Error: 3 required arguments <parser model> <word dictionary> <input conll file>\n")
        sys.exit(-1)


    logging.basicConfig(level=logging.INFO)
    logging.info("Reading parser models")
    in_file = open(args[0], 'rb')
    models = pickle.load(in_file)
    parser = HmmParser.HmmParser(models)
    
    logging.info("Reading word->int mapping dictionary")

    # Read in dictionary so we can map input to integers for parser
    f = open(args[1], 'r')
    word_map = {}
    word_lookup = {}
    for line in f:
        #pdb.set_trace()
        (word, index) = line.rstrip().split(" ")
        word_map[word] = int(index)
        word_lookup[index] = word

    logging.info("Parsing sentences...")
    
    with open(args[2], 'r') as f:
        lines = []
        tokens = []
        int_tokens = []
        for line in f:
            fields = line.strip().split('\t')
            if line[0] == '#':
                print(line.strip())
            elif len(fields) < 2:
                ## Empty line -- collect sentence and parse:
                for token in tokens:
                    if token in word_map:
                        int_tokens.append(word_map[token])
                    else:
                        int_tokens.append(word_map['unk'])
                parse = parser.matrix_parse(int_tokens)
                for index, state in enumerate(sent_list):
                    print("Index=%d, state=%s" % (index, state.str() ) )
                    
            elif len(fields) > 1:
                tokens.append(fields[1].lower())
                #tokens.append(line.strip().split('\t')[1].lower())
            else:
                print("Didn't know what to do with this line: %s" % (line) )

if __name__ == "__main__":
    main(sys.argv[1:])

