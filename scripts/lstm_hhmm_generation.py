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

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
text = open('data/simplewiki_d1.words.lc.txt').read()
tokens = re.compile('\s+').split(text)
print('corpus length:', len(tokens))

types = sorted(list(set(tokens)))
print('total tokens:', len(types))
token_indices = dict((c, i) for i, c in enumerate(types))
indices_tokens = dict((i, c) for i, c in enumerate(types))
fn = open('hhmm_alphabets.pkl', 'w')
pickle.dump( (token_indices, indices_tokens), fn)
fn.close()

# cut the text in semi-redundant sequences of maxlen tokens
#maxlen = 20
#step = 3
sentences = []
next_tokens = []

for line in text.split('\n'):
    tokens = line.split(' ')
    if len(tokens) < 3:
        continue
    sentences.append([token_indices[x] for x in tokens])

print('nb sequences:', len(sentences))

print('Vectorization...')
X = pad_sequences(sentences)
maxlen = X.shape[1]
y = np.zeros((len(sentences), maxlen, len(types)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    next_words = [] #[False] * len(X[i])
    for j in range(0, len(sentences[i])-1):
#        next_words.append( [False] * len(types) )
#        next_words[-1][sentences[i][j+1]] = True
        y[i, j, X[i,j+1]] = 1

#    y.append(next_words)

print('Build model...')
embed_dim = 30
syn_dim = 20
input_layer = Input(shape=(None,), dtype='int32')
embed = Embedding(input_dim=len(types), output_dim=embed_dim)(input_layer)
hhmm = HHMMLayer(embed_dim=embed_dim, syn_dim=syn_dim, return_sequences=True)(embed)

#output = Dense(len(types), activation='softmax')(hhmm)
output = TimeDistributed(Dense(len(types), activation='softmax'))(hhmm)

optimizer = RMSprop(lr=0.001)
model = Model(input = input_layer, output = output)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

## Write model structure (does not require training and doeesn't change between iterations)
json_string = model.to_json()
open('hhmm_model.json', 'w').write(json_string)


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
    print('Iteration', iteration)
    model.fit(X, y, batch_size=64, nb_epoch=1)

    model.save_weights('hhmm_model.h5', overwrite=True)

    start_index = 0 #random.randint(0, len(tokens) - maxlen - 1)
    sent_to_complete_ind = random.randint(0, X.shape[0])
    sent_to_complete = sentences[sent_to_complete_ind]
    sent_len = len(sent_to_complete)

    end_index = np.random.randint(min(3, sent_len-1), sent_len)
    sub_sent = sent_to_complete[0:end_index]


    for diversity in [1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        
        sentence = sub_sent
        sentence_str = ' '.join([indices_tokens[x] for x in sub_sent]) + ' '
        #print('Source sentence: "%s"' % sent_to_complete)
        #print('Sub-sentence initialization: "%s"' % sub_sent)
        generated += sentence_str
        print('----- Generating with seed: "' + sentence_str + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1,maxlen))
            for t, token in enumerate(sentence):
                x[0,t] = token

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds[-1], diversity)
            next_token = indices_tokens[next_index]

            generated += next_token + ' '
            sentence.append(next_index)
            sentence = sentence[1:]

            sys.stdout.write(next_token)
            sys.stdout.write(' ')
            sys.stdout.flush()
        print()
