#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import glob
import pickle
import HmmParser
import sys
from ihmm import State
import logging

def main(args):
    if len(args) < 4:
        sys.stderr.write("Error: 4 required arguments <parser model> <word dictionary> <ltf directory> <Output directory>\n")
        sys.exit(-1)
        
    logging.basicConfig(level=logging.INFO)
    logging.info("Reading parser models")
    in_file = open(args[0], 'rb')
    models = pickle.load(in_file)
    parser = HmmParser.HmmParser(models)
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
    for ltf_file in files:
        sys.stderr.write("Processing file %s\n" % ltf_file)
        tree = ET.parse(ltf_file)
        root = tree.getroot()
        doc = root.find("DOC")
        docid = doc.get("id")
        
        out_root = ET.Element("LCTL_ANNOTATIONS")
        out_doc = ET.SubElement(out_root, "DOC")
        out_doc.set("id", docid)
        annot_id = 0
        
        for seg in root.iter('SEG'):
            cur_sent = list()
            token_ids = []
            token_starts = []
            token_ends = []
            for token in seg.iter('TOKEN'):
                tag = token.attrib.get('pos')
                if tag == None:
                    tag = 'UNK'
                    #cur_sent = None
                    #break
                
                tag = tag.replace('/', '-slash-')
                word = token.text
                if word == None or len(word) == 0:
                    logging.warning("Found an empty token!")
                    continue
                cur_sent.append(word.lower())
                token_ids.append(token.get("id"))
                token_starts.append(token.get("start_char"))
                token_ends.append(token.get("end_char"))
            
            int_tokens = []
            for token in cur_sent:
                if token in word_map:
                    int_tokens.append(word_map[token])
                else:
                    int_tokens.append(word_map['unk'])
            
            if len(int_tokens) > 0:
                sent_list = parser.parse(int_tokens)
                #print("Received output with %d states" % (len(sent_list)))            
                #print(list(map(lambda x: x.str(), sent_list) ) )

                ## Extent for root of tree                
                add_annotation(out_doc, cur_sent, docid + "-ann-" + str(annot_id), token_ids[0], token_ids[-1], token_starts[0], token_ends[-1])
                annot_id += 1

                marker = 0
                for index,state in enumerate(sent_list):
                    if index == 0:
                        continue
                    
                    if state.f == 0 and state.j == 0:
                        ## Everything we've seen since the last marker is a span
                        add_annotation(out_doc, cur_sent[marker:index], docid + "-ann-" + str(annot_id), token_ids[marker], token_ids[index], token_starts[marker], token_ends[index])
                        annot_id += 1
                        #print("Found minor span from %d to %d with string %s" % (marker, index, " ".join(cur_sent[marker:index])) )
                        if marker > 0:
                            ## If we split in the middle of the sentence we need to join the two split sections.
                            add_annotation(out_doc, cur_sent[0:index], docid + "-ann-" + str(annot_id), token_ids[0], token_ids[index], token_starts[0], token_ends[index])
                            annot_id += 1
                            #print("Found major span from %d to %d with string %s" % (0, index, " ".join(cur_sent[0:index]) ) )
                        marker = index
            
                ## And one last span at the end:
                if marker > 0:
                    add_annotation(out_doc, cur_sent[marker:-1], docid + "-ann-" + str(annot_id), token_ids[marker], token_ids[-1], token_starts[marker], token_ends[-1])
                    annot_id += 1
        indent(out_root)
        out_tree = ET.ElementTree(out_root)
        out_tree.write('%s/%s.xml' % (out_dir, docid), encoding='utf-8')

def add_annotation(parent, list, id_str, begin_id, end_id, begin_char, end_char):
    annot = ET.SubElement(parent, "ANNOTATION")
    annot.set("id", id_str)
    annot.set("task", "NE")
    annot.set("start_token", begin_id)
    annot.set("end_token", end_id)
    extent = ET.SubElement(annot, "EXTENT")
    extent.set("start_char", begin_char)
    extent.set("end_char", end_char)
    extent.text = " ".join(list)

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
