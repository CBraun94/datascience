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
    ax.plot(r, inertias, marker='o')
    ax.set_title('Elbow method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    fig.savefig(fname=DIR_OUT+'kmeans_elbow')
    #plt.close()


def plot_sil(l_index: list, sil_score: list, fig=None, ax=None):
    plt.style.use(PLT_STYLE)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(l_index, sil_score, marker='o')
    ax.set_title('silhouette_score')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('silhouette_score')
    ax.set_xticks(l_index)
    fig.savefig(fname=DIR_OUT+'sil_score')
    #plt.close()


def plot_scatter(color, x, y, filename: str, fig=None, ax=None, title=None, xlabel=None, ylabel=None):
    plt.style.use(PLT_STYLE)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, c=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    if filename is not None and filename != '':
        fig.savefig(fname=DIR_OUT+filename)


def plot_bar(x, y, fig=None, ax=None):
    plt.style.use(PLT_STYLE)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    #ax.bar(x=x, height=y, width=0.01)
    aaa = list(map(str, y))
    ax.pie(x, labels=aaa)


def plot_analyze_sil(df_k, df: pl.DataFrame, df_data_clustered: pl.DataFrame, kmeans, filename):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    plt.style.use(PLT_STYLE)
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('plot_analyze_sil')

    gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.5)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots()

    _x = df_data_clustered.to_series(df_data_clustered.get_column_index('Source_One')).to_list()
    _y = df_data_clustered.to_series(df_data_clustered.get_column_index('Source_Two')).to_list()
    _z = df_data_clustered.to_series(df_data_clustered.get_column_index('Source_Three')).to_list()

    _color = df_data_clustered.to_series(df_data_clustered.get_column_index('cluster')).to_list()
    _inertias = df_k.to_series(df_k.get_column_index('inertias')).to_list()
    _k = df_k.to_series(df_k.get_column_index('k')).to_list()
    _sil_score = df_k.to_series(df_k.get_column_index('sil_score')).to_list()

    df_count = df_data_clustered.sort('cluster').group_by('cluster').count()
    plot_bar(x=df_count.to_series(df_count.get_column_index('count')).to_list(), y=df_count.to_series(df_count.get_column_index('cluster')).to_list(), fig=fig, ax=ax3)

    plot_elbow(inertias=_inertias, fig=fig, ax=ax1)
    plot_sil(l_index=_k, sil_score=_sil_score, fig=fig, ax=ax2)

    plot_scatter(color=_color, x=_x, y=_y, filename='xy', fig=fig, ax=ax4, title='xy', xlabel='Source_One', ylabel='Source_Two')
    plot_scatter(color=_color, x=_x, y=_z, filename='xz', fig=fig, ax=ax5, title='xz', xlabel='Source_One', ylabel='Source_Three')
    plot_scatter(color=_color, x=_z, y=_y, filename='zy', fig=fig, ax=ax6, title='zy', xlabel='Source_Three', ylabel='Source_Two')

    # fig.tight_layout()
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

    if filename is not None and filename != '':
        fig.savefig(fname=DIR_OUT+filename)

    plt.close(fig)
