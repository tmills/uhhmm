#!/usr/bin/env python3

import sys

from gensim.models.keyedvectors import KeyedVectors

from scipy.cluster.vq import kmeans

def main(args):
    if len(args) < 1:
        sys.stderr.write("One required argument: <embeddings file>\n")
        sys.exit(-1)

    print("Reading embeddings: ")
    embeddings = KeyedVectors.load_word2vec_format(args[0], binary=False)
    means = embeddings.syn0.mean(0)
    stdevs = embeddings.syn0.std(0)
    mean_max = means.max()
    mean_min = means.min()
    mean_mean = means.mean()
    mean_stdev = means.std()
    stdevs_mean = stdevs.mean()
    stdevs_stdev = stdevs.std()

    print('mean stats: mean=%f, stdev=%f, max=%f, min=%f' % (mean_mean, mean_stdev, mean_max, mean_min))
    print('stdev stats: mean=%f, stdev=%f' % (stdevs_mean, stdevs_stdev))

    centroids, distortion = kmeans(embeddings.syn0, 10, iter=5)

    print("K means centroids: %s" % (str(centroids)))

if __name__ == "__main__":
    main(sys.argv[1:])
