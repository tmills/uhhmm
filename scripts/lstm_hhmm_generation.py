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
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import re
from HHMM import HHMMLayer

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
tokens = re.compile('\s+').split(text)
print('corpus length:', len(tokens))

types = sorted(list(set(tokens)))
print('total tokens:', len(types))
token_indices = dict((c, i) for i, c in enumerate(types))
indices_tokens = dict((i, c) for i, c in enumerate(types))

# cut the text in semi-redundant sequences of maxlen tokens
maxlen = 10
step = 3
sentences = []
next_tokens = []
for i in range(0, len(tokens) - maxlen, step):
    sentences.append(tokens[i: i + maxlen])
    next_tokens.append(tokens[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen), dtype=np.bool)
y = np.zeros((len(sentences), len(types)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, token in enumerate(sentence):
        X[i, t] = token_indices[token]
    y[i, token_indices[next_tokens[i]]] = 1


# build the model: 2 stacked SimpleRNN
print('Build model...')
embed_dim = 100
syn_dim = 50
input = Input(shape=(maxlen,), dtype='int32')
embed = Embedding(input_dim=len(types), output_dim=embed_dim)(input)
hhmm = HHMMLayer(embed_dim=embed_dim, syn_dim=syn_dim)(embed)

output = Dense(len(types), activation='softmax')(hhmm)

optimizer = RMSprop(lr=0.01)
model = Model(input = input, output = output)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(tokens) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = tokens[start_index: start_index + maxlen]
        sentence_str = ' '.join(sentence)
        generated += sentence_str
        print('----- Generating with seed: "' + sentence_str + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(types)))
            for t, token in enumerate(sentence):
                x[0, t, token_indices[token]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_token = indices_tokens[next_index]

            generated += next_token + ' '
            sentence.append(next_token)
            sentence = sentence[1:]

            sys.stdout.write(next_token)
            sys.stdout.write(' ')
            sys.stdout.flush()
        print()
