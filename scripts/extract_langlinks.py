#!/usr/bin/env python3.4

import sys

def read_ner_lines(fn):
    ## Open file, replace spaces in titles with underscores, create map from title to 
    ## NER type
    map = {}
    
    ## code here
    ner_file = open(fn, 'r')
    for line in ner_file:
        parts = line.rstrip().split(' ')
        key = "_".join(parts[1:])
        map[key] = parts[0]
        sys.stderr.write("Creating map with %s => %s\n" % (key, parts[0]) )

#    map["Barack_Obama"] = "PER"
    return map
    
def main(args):
    if len(args) < 2:
        print "ERROR: 2 required arguments: <formatted wiki page file> <formatted wiki lang dump> <ner prototypes>"
        sys.exit(-1)
        
    wiki_page_file = args[0]
    wiki_lang_file = args[1]
    ner_file = args[2]
    
    ners = read_ner_lines(ner_file)
    
    ## Find the lines in the page file with the NER titles
    page_maps = {}
    pf = open(wiki_page_file, 'r')
    for line in pf:
        if not line.startswith("("):
            continue
        els = line.split(',')
        namespace_id = int(els[1])
        if not namespace_id == 0:
            continue
        pid = els[0][1:]
        name_with_quotes = els[2]
        name_wo_quotes = name_with_quotes[1:-1]
        if name_wo_quotes in ners:
            sys.stderr.write("Found key %s in page file with pid %d\n" % (name_wo_quotes, int(pid)) )
            page_maps[pid] = ners[name_wo_quotes]
    
    pf.close()
    
    ## Now find the other language links:
    lf = open(wiki_lang_file, 'r')
    for line in lf:
        if not line.startswith("("):
            continue
        
        line = line.rstrip()
        els = line[1:-1].split(',')
        pid = els[0]
        lang_code = els[1][1:-1]
        title = els[2][1:-1]
        if pid in page_maps:
            ## Print out the NER type, the language code, and the term
            print("%s %s %s" % (page_maps[pid], lang_code, title) )
    
if __name__ == "__main__":
    main(sys.argv[1:])
