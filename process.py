import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.cluster.hierarchy as sch
import pandas as pd


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def readPd(root):
    df = pd.read_csv(root, sep='\t')
    return df.T


def get_train_genes(fname):
    df = pd.read_csv(fname, dtype=str)
    gene_id = df.iloc[:, 0].astype(str)
    return pd.unique(gene_id)


train_root = 'data/E003/classification/train.csv'
gene_id = get_train_genes(train_root)
path = '57epigenomes.RPKM.pc'

lst = readPd(path)[2:-1]
smallest_key = len(min(gene_id, key=len))
bool_pd = np.isin(lst.columns.str[-smallest_key:], gene_id)
train_pd = lst.T[bool_pd]

corr_pd = cluster_corr(train_pd.corr())
corr = np.array(corr_pd)
np.savez("correlation_matrix.npz", corr)
corr_pd.to_csv("correlation_pd.csv")
figure, ax = plt.subplots(1, 2, figsize=(56, 20))
for i in range(len(lst)):
    one_corr = corr[i, :]
    one_corr.sort()
    one_corr = one_corr[::-1]
    ax[0].plot(np.arange(len(lst)), one_corr)
sn.heatmap(corr_pd, xticklabels=True, yticklabels=True, cmap="seismic")

figure.savefig("Correlation" + path[-2:] + ".png")
