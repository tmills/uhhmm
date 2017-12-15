'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Embedding
from keras.layers import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import random
import sys
import re
from HHMM import HHMMLayer

embed_dim = 30
syn_dim = 20
max_batch = 128

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
text = open('data/simplewiki_d2_all.words.lc.txt').read()
tokens = re.compile('\s+').split(text.strip())
print('corpus length:', len(tokens))

empty_token = "<empty>"
start_token = "<s>"
end_token = "</s>"

types = [empty_token, start_token, end_token]
types.extend(sorted(list(set(tokens))))

print('total tokens:', len(types))
token_indices = dict((c, i) for i, c in enumerate(types))
indices_tokens = dict((i, c) for i, c in enumerate(types))
fn = open('rhhmm_alphabets.pkl', 'w')
pickle.dump( (token_indices, indices_tokens), fn)
fn.close()

# cut the text in semi-redundant sequences of maxlen tokens
#maxlen = 20
#step = 3
sents_by_len = {}
all_sentences = []
next_tokens = []

lines = text.strip().split('\n')
for line in lines:
    tokens = [start_token]
    tokens.extend(line.strip().split(' '))
    #tokens.append(end_token)
    if len(tokens) < 3:
        continue
    
    int_tokens = [token_indices[x] for x in tokens]
    all_sentences.append(int_tokens)
    
    if not len(tokens) in sents_by_len:
        sents_by_len[ len(tokens) ] = []
        
    sents_by_len[ len(tokens) ].append(int_tokens)

print('nb sequences:', len(all_sentences))

print('Build model...')

input_layer = Input(shape=(None,), dtype='int32')
embed = Embedding(input_dim=len(types), output_dim=embed_dim)(input_layer)
hhmm = HHMMLayer(embed_dim=embed_dim, syn_dim=syn_dim, return_sequences=True)(embed)

output = TimeDistributed(Dense(len(types), activation='softmax'))(hhmm)

optimizer = RMSprop(lr=0.0001)
model = Model(input = input_layer, output = output)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

## Write model structure (does not require training and doeesn't change between iterations)
json_string = model.to_json()
open('rhhmm_model.json', 'w').write(json_string)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 500):
    print()
    print('-' * 50)
    print('Iteration %d' % iteration)
    
    epoch_loss = 0
    for sent_len in sents_by_len.keys():
        sentences = sents_by_len[sent_len]
        print("Training on %d sentences of length %d" % (len(sentences), sent_len) )
        
        batch_start = 0
        
        while batch_start < len(sentences):
            end_ind = min(batch_start + max_batch, len(sentences))
            X = pad_sequences(sentences[batch_start:end_ind])  ## shouldn't actually be padded but will be np array
            Y = np.zeros((X.shape[0], sent_len, len(types)), dtype=np.bool)
            for i in range(X.shape[0]):
                sentence = X[i]
                next_words = [] #[False] * len(X[i])
                for j in range(0, sent_len-1):
                    Y[i, j, X[i,j+1]] = 1
                Y[i, -1, token_indices[end_token] ] = 1
            
            ## Train on this batch:
            #print("Training on batch from index %d to %d" % (batch_start, end_ind) )
            epoch_loss += model.train_on_batch(X, Y)
            batch_start += max_batch
            
        if np.isnan(epoch_loss):
            sys.stderr.write("Exiting since training error is now NaN!\n")
            sys.exit(-1)
#    model.fit(X, y, batch_size=64, nb_epoch=1)
    
    print("Total loss for this pass through data: %f"  % (epoch_loss) )
    
    model.save_weights('rhhmm_model.h5', overwrite=True)

    start_index = 0 #random.randint(0, len(tokens) - maxlen - 1)
    sent_to_complete = random.choice(all_sentences)
    sent_len = len(sent_to_complete)

    end_index = np.random.randint(min(3, sent_len-1), sent_len)
    sub_sent = sent_to_complete[0:end_index]


    for diversity in [1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        
        sentence = sent_to_complete[0:end_index]
        sentence_str = ' '.join([indices_tokens[x] for x in sub_sent]) + ' '
        #print('Source sentence: "%s"' % sent_to_complete)
        #print('Sub-sentence initialization: "%s"' % sub_sent)
        generated += sentence_str
        print('----- Generating with seed: "' + sentence_str + '"')
        sys.stdout.write(generated)

        for i in range(10):
            x = np.zeros((1,len(sentence)))
            for t, token in enumerate(sentence):
                x[0,t] = token

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds[-1], diversity)
            next_token = indices_tokens[next_index]
                

            generated += next_token + ' '
            sentence.append(next_index)
                
            #sentence = sentence[1:]

            sys.stdout.write(next_token)
            sys.stdout.write(' ')
            
            if next_token == end_token:
                sentence.append(token_indices[start_token])
                sys.stdout.write(start_token)
                sys.stdout.write(' ')
                
            sys.stdout.flush()
        print()

