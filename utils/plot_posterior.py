import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas, scipy.stats
import sys, os
# import matplotlib.colors as mcolors
#
# colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
# cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
# colors = [(0,0,1,c) for c in np.linspace(0,1,100)]
# cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
def limit_array(arr, r_limit=300):
    arr *= (300/arr.max())
    return arr

def burn_in_removal(arr, burn_in=10):
    return arr[burn_in:]

def plot_multiple_chains(sample_chains, logprobs, burn_in=100):
    """
    to plot a scatter plot of the posterior using dimensionality reduction techniques
    the distributions are first PCA-ed and then TSNE-ed into 2 dimensional data
    :param sample_chains: a list of numpy arrays of size (iters, params)
    :param logprobs: a list of logprobs corresponding to the samples (iters,)
    :return: a pdf file of the posterior
    """
    # preprocessing
    sample_chains = [burn_in_removal(x, burn_in) for x in sample_chains]
    logprobs = [limit_array(burn_in_removal(x, burn_in)) for x in logprobs]

    num_sample_list = [x.shape[0] for x in sample_chains]
    data_matrix = np.vstack(sample_chains)
    # dim reduction
    reduced_samples = PCA(n_components=200).fit_transform(data_matrix)
    # print(reduced_samples)
    bidigit_data = reduced_samples

    tsne_model = TSNE(n_components=2, learning_rate=1000, metric='euclidean', perplexity=50)
    # tsne_model = TSNE(n_components=2, learning_rate=1000, metric=scipy.stats.entropy, perplexity=50)
    bidigit_data = tsne_model.fit_transform(bidigit_data)
    # print(bidigit_data)

    # generate the colors
    total_samples = sum(num_sample_list)
    num_chains = len(sample_chains)
    colors_per_point = np.zeros(total_samples)
    color_map = plt.get_cmap('jet')
    cur_index = 0
    running_sum = num_sample_list[0]
    for i in range(total_samples):
        if i >= running_sum:
            cur_index += 1
            running_sum += num_sample_list[cur_index]
        colors_per_point[i] = cur_index / num_chains
    data_point_sizes = np.vstack(logprobs)

    fig, ax = plt.subplots()
    points = ax.scatter(bidigit_data[:, 0], bidigit_data[:, 1], c=colors_per_point, s=data_point_sizes, marker='.', cmap=color_map
                        , alpha=0.1)
    ax.set_ylabel('Components')
    # ax.legend(points, ("D2 trees"))
    pp = PdfPages('chains_posterior' + '.pdf')
    fig.savefig(pp, format='pdf')
    pp.close()
    plt.cla()
    plt.clf()

if __name__ == '__main__':
    # chains = [np.random.normal(5, 1, (25, 1000)), np.random.normal(-1, 2, (25, 1000))]
    # print(chains)
    working_dir = sys.argv[1]
    nonterms = pandas.read_table(os.path.join(working_dir,'pcfg_nonterms.txt')).as_matrix()
    terms = pandas.read_table(os.path.join(working_dir,'pcfg_terms.txt')).as_matrix()
    allterms = np.hstack((terms,nonterms))
    chains = [allterms]
    logprobs = [pandas.read_table(os.path.join(working_dir,'pcfg_hypparams.txt')).logprob.as_matrix()]
    plot_multiple_chains(chains, logprobs)