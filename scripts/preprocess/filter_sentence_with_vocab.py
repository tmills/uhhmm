#!/usr/bin/env python3.4

import sys

def main(args):
    words = get_word_set_from_file(args[0])
    sentences = get_sentence_list_from_file(args[1])
    
    filtered = get_filtered_sentence_indices(sentences, words)
    
    for ind in filtered:
        print(sentences[ind])
    
def get_word_set_from_file(filename):
    f = open(filename, 'r')
    word_dict = {}
    for line in f:
        word = line.rstrip()
        word_dict[word] = 1
    f.close()
    return word_dict
    
def get_sentence_list_from_file(filename):
    f = open(filename, 'r')
    sents = []
    for line in f:
        sent = line.rstrip()
        sents.append(sent)
    f.close()
    return sents

def get_filtered_sentence_indices(sents, words):
    passing_inds = []
    passes = True
    
    for index,sent in enumerate(sents):
        passes = True
        word_list = sent.split()
        for token in word_list:
            if not token in words:
                ## This sentence has a token not in our dictionary
                passes = False
                break

        if passes:        
            ## If we get here our sentence passed
            passing_inds.append(index)
    
    return passing_inds
    
if __name__ == "__main__":
    main(sys.argv[1:])
