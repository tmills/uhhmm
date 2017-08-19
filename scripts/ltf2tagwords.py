import xml.etree.ElementTree as ET
import glob
import sys

def main(args):
    if len(args) == 3:
        outputfile = open(args[1], 'w')
        outindexfile = open(args[2], 'w')
    if sys.version_info[0] != 3:
        print("This script requires Python 3")
        exit()
    
    files = glob.glob(args[0] + "/*.ltf.xml")
    for ltf_file in files:
        sys.stderr.write("Processing file %s\n" % ltf_file)
        try:
            tree = ET.parse(ltf_file)
            root = tree.getroot()
            for doc in root.iter('DOC'):
                cur_doc = doc.attrib.get('id')
                for seg in root.iter('SEG'):
                    cur_seg = seg.attrib.get('id')
                    cur_sent = list()
                    for token in seg.iter('TOKEN'):
                        tag = token.attrib.get('pos')
                        if tag == None:
                            tag = 'UNK'
                            #cur_sent = None
                            #break

                        tag = tag.replace('/', '-slash-')
                        word = token.text
                        cur_sent.append("%s/%s" % (tag, word))

                    if cur_sent != None:
                        if len(args) == 3:
                            print(' '.join(cur_sent), file=outputfile)
                            print(' '.join([ltf_file, cur_doc, cur_seg]), file=outindexfile)
                        else:
                            print(' '.join(cur_sent))
        except:
            sys.stderr.write("Error parsing file %s\n" % ltf_file)

if __name__ == "__main__":
    main(sys.argv[1:])

