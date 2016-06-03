#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import glob
import pickle
import HmmParser
import sys
from State import State
import logging
import multiprocessing



def main(args):
    if len(args) < 4:
        sys.stderr.write("Error: 4 required arguments <parser model> <word dictionary> <ltf directory> <Output directory>\n")
        sys.exit(-1)
        
    logging.basicConfig(level=logging.INFO)
    logging.info("Reading parser models")
    in_file = open(args[0], 'rb')
    models = pickle.load(in_file)
    out_dir = args[3]
    
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

    logging.info("Parsing ltf files")
    files = glob.glob(args[2] + "/*.ltf.xml")

#    jobs = []
    pool = multiprocessing.Pool(processes=10)

    for ltf_file in files:
        pool.apply_async(processLTF, (ltf_file, models, word_map, word_lookup, out_dir))
    pool.close()
    pool.join()
#        p = multiprocessing.Process(target=processLTF, args=(ltf_file, parser, word_map, word_lookup, out_dir))
#        jobs.append(p)
#        p.start()



def processLTF(ltf_file, models, word_map, word_lookup, out_dir):
    parser = HmmParser.HmmParser(models)
    sys.stderr.write("Processing file %s\n" % ltf_file)
    tree = ET.parse(ltf_file)
    root = tree.getroot()
    doc = root.find("DOC")
    docid = doc.get("id")

    for seg in root.iter("SEG"):
        cur_sent = []
        token_ids = []
        token_elemsByID = {}
        for token in seg.iter("TOKEN"):
            word = token.text
            if word == None or len(word) == 0:
                logging.warning("Found an empty token!")
                continue
            cur_sent.append(word.lower())
            token_ids.append(token.get('id'))
            token_elemsByID[token.get('id')] = token

        int_tokens = []
        for token in cur_sent:
            if token in word_map:
                int_tokens.append(word_map[token])
            else:
                int_tokens.append(word_map['unk'])

        if len(int_tokens) > 0:
#            token_ids.insert(0, None)
            sent_list = parser.parse(int_tokens)
            for index, state in enumerate(sent_list):

#                if index == 0:
#                    continue

                # add a, b, and g attributes to token elements of existing tree
                # provided the states in sent_list and the tokens in token_ids
                # line up 1 to 1

                addABG2TokenElement(token_elemsByID[token_ids[index]], state)

    tree.write("%s/%s.ltf.xml" % (out_dir, docid), encoding='utf-8')



def addABG2TokenElement(element, state):
    element.set('a', str(state.a))
    element.set('b', str(state.b))
    element.set('g', str(state.g))
    element.set('f', str(state.f))
    element.set('j', str(state.j))



def indent(elem, level=0):
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem




if __name__ == "__main__":
    main(sys.argv[1:])
