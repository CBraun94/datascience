import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import polars as pl


DIR_OUT = r'data/output/'


PLT_STYLE = 'dark_background'


def plot_elbow(inertias: list, fig=None, ax=None):
    r = range(1, len(inertias)+1)
    plt.style.use(PLT_STYLE)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    plt.plot(r, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(fname=DIR_OUT+'kmeans_elbow')
    plt.close()


def plot_sil(l_index: list, sil_score: list, fig=None, ax=None):
    plt.style.use(PLT_STYLE)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    plt.plot(l_index, sil_score, marker='o')
    plt.title('silhouette_score')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.xticks(l_index)
    plt.savefig(fname=DIR_OUT+'sil_score')
    plt.close()


def plot_scatter(kmeans, x, y, filename: str, fig=None, ax=None, title=None, xlabel=None, ylabel=None):
    plt.style.use(PLT_STYLE)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, c=kmeans.labels_)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    if filename is not None and filename != '':
        fig.savefig(fname=DIR_OUT+filename)


def plot_analyze_sil(df: pl.DataFrame, df_data_clustered: pl.DataFrame):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    filename = 'plot_analyze_sil'
    plt.style.use(PLT_STYLE)
    fig = plt.figure()
    fig.suptitle('plot_analyze_sil')

    gs = fig.add_gridspec(2, 3)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots()

    if filename is not None and filename != '':
        fig.savefig(fname=DIR_OUT+filename)
