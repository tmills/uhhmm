import sys
import os
import logging
from os.path import join
#from lxml import etree as ET
import xml.etree.ElementTree as ET

#data = "We welcome the President of the United States of America , whose contributions have been great for America .".split()

#gaz = ["president of the united states of america"
#        ,"united states of america"
#        ,"united states"
#        ,"america"
#        ]

#g1 = set([entry.split()[0] for entry in gaz])

class Candidate:
    def __init__(self, word, start_char, end_char, start_token, end_token):
        self.string = word
        self.lowered = word.lower()
        self.start_char = start_char
        self.end_char = end_char
        self.start_token = start_token
        self.end_token = end_token

    def appendString(self, word, new_end_char, new_end_token):
        self.string+=' '+word
        self.lowered+=' '+word.lower()
        self.end_char = new_end_char
        self.end_token = new_end_token


def main(args):
    """
    arg0 = gazetteer file
    arg1 = ltf directory
    arg2 = laf output directory

    """
    try:
        gaz_fn = args[0]
        ltf_dir = args[1]
        laf_dir = args[2]
        working_lang = args[3]

    except:
        print("Usage: "+sys.argv[0]+" [gazetteer file] [ltf directory] [laf output directory] [lang abbrev]")
        exit(1)

    logging.basicConfig(level=logging.INFO)
#    logging.basicConfig(level=logging.DEBUG)

    with open(gaz_fn, 'r') as gazfile:
        gaz = [] # List of full entries in gazetteer
        g1 = [] # List of first words of entries in gazetteer
        entitytypes = {}
        for line in gazfile.read().strip().split('\n'):
            NEtype = line.split()[0]
            lang = line.split()[1]
            entity = ' '.join(line.split()[2:])
            if lang == working_lang:
                entitytypes[entity] = NEtype
                try:
                    u_entity = entity.lower()
                    u_entity_start = entity.lower()
                    gaz.append(u_entity)
                    g1.append(u_entity_start)
                    print("Added %s of type %s to gazetteer" % (u_entity, type(u_entity) ) )
                except Exception as e:
                    print ("Failed to encode this entity: %s" % entity)
                                    
    for fn in os.listdir(args[1]):
        if fn.endswith("ltf.xml"):
            annot_id = 0
            tree = ET.parse(join(ltf_dir, fn))
            root = tree.getroot()
            doc = root.find("DOC")
            doc_id = doc.get("id")
            out_root = ET.Element("LCTL_ANNOTATIONS")
            out_doc = ET.SubElement(out_root, "DOC")
            out_doc.set("id", doc_id)
            for seg in root.iter('SEG'):
                candidates = []
                for token in seg.iter('TOKEN'):
                    word = token.text
                    if word == None or len(word) == 0:
                        logging.warning("Found an empty token!")
                        continue
                    if type(word) == 'unicode':
                        print("re-encoding unicode word %s" % (word) )
                        word = word.decode('utf-8').encode('utf-8')
                        
                    logging.debug("Processing %s" % word)
                    token_id = token.get('id')
                    start_char = token.get('start_char')
                    end_char = token.get('end_char')
                    candidate = None
                    #print("Type of word %s is %s, type of g1 is %s" % ( word.lower(), type(word.lower()), type(g1[0]) ) )
                    if word.lower() in g1: # If the current word is the first of a NE, start a new candidate with this word
                        logging.debug("Found first word of NE")
                        candidate = Candidate(word, start_char, end_char, token_id, token_id)
                    for c in candidates: # Add the current word to all already started candidates, and see if there are any complete NE matches
                        c.appendString(word, end_char, token_id)
                        if c.lowered in gaz:
                            logging.debug("Found full NE in gazetteer")
                            add_annotation(out_doc, c.string, doc_id+'-ann-'+str(annot_id), entitytypes[c.string],
                                            c.start_token, c.end_token, c.start_char, c.end_char)
                            annot_id+=1
                    if candidate: # If a new candidate was started, check whether it is a complete NE, and add it to the list of candidates
                        if candidate.lowered in gaz:
                            logging.debug("Found full NE in gazetteer")
                            add_annotation(out_doc, candidate.string, doc_id+'-ann-'+str(annot_id), entitytypes[candidate.string],
                                            candidate.start_token, candidate.end_token, candidate.start_char, candidate.end_char)
                            annot_id+=1
                        candidates.append(candidate)

        indent(out_root)
        out_tree = ET.ElementTree(out_root)
        out_tree.write(join(laf_dir, fn.replace('ltf.xml','laf.xml')))


def add_annotation(parent, string, id_str, NEtype, begin_id, end_id, begin_char, end_char):
    annot = ET.SubElement(parent, "ANNOTATION")
    annot.set("id", id_str)
    annot.set("task", "NE")
    annot.set("start_token", begin_id)
    annot.set("end_token", end_id)
    annot.set("type", NEtype)
    extent = ET.SubElement(annot, "EXTENT")
    extent.set("start_char", begin_char)
    extent.set("end_char", end_char)
    extent.text = string


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



if __name__ == '__main__':
    main(sys.argv[1:])    
