import matplotlib.pyplot as plt
import polars as pl
import _const as _c
from sklearn.cluster import KMeans

PLT_STYLE = 'dark_background'


def plot_elbow(inertias: list, fig=None, ax=None):
    r = range(1, len(inertias)+1)
    plt.style.use(PLT_STYLE)
    _close: bool = False
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        _close = True
    ax.plot(r, inertias, marker='o')
    ax.set_title('Elbow method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    fig.savefig(fname=_c.DIR_OUT+'kmeans_elbow')

    if _close:
        plt.close(fig=fig)


def plot_sil(l_index: list, sil_score: list, fig=None, ax=None):
    plt.style.use(PLT_STYLE)
    _close: bool = False
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        _close = True
    ax.plot(l_index, sil_score, marker='o')
    ax.set_title('silhouette_score')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('silhouette_score')
    ax.set_xticks(l_index)
    fig.savefig(fname=_c.DIR_OUT+'sil_score')

    if _close:
        plt.close(fig=fig)


def plot_scatter(color, x, y, filename: str, fig=None, ax=None, title=None, xlabel=None, ylabel=None):
    plt.style.use(PLT_STYLE)
    _close: bool = False
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        _close = True

    ax.scatter(x, y, c=color, cmap='Accent', alpha=_c.ALPHA)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    if filename is not None and filename != '':
        fig.savefig(fname=_c.DIR_OUT+filename)

    if _close:
        plt.close(fig=fig)


def plot_bar(x, y, fig=None, ax=None):
    plt.style.use(PLT_STYLE)
    _close: bool = False
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        _close = True
    #ax.bar(x=x, height=y, width=0.01)
    #aaa = list(map(str, y))
    ax.pie(x, labels=y)

    if _close:
        plt.close(fig=fig)


def plot_scatter3d(x, y, z, color, filename: str, fig=None, ax=None):
    plt.style.use(PLT_STYLE)
    _close: bool = False
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        _close = True

    ax.scatter(x, y, z, c=color, cmap='Accent', alpha=_c.ALPHA)

    if _c.INTERACTIVE:
        plt.show()
        fig.show()

    if filename is not None and filename != '':
        fig.savefig(fname=_c.DIR_OUT+filename)

    #if _close:
        #plt.close(fig=fig)


def plot_analyze_sil(df_k, df: pl.DataFrame, df_data_clustered: pl.DataFrame, kmeans: KMeans, filename, index):
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    plt.style.use(PLT_STYLE)
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(str(index) + ' Cluster Analysis')

    gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.5)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots()

    _x = df_data_clustered.to_series(df_data_clustered.get_column_index('Source_One')).to_list()
    _y = df_data_clustered.to_series(df_data_clustered.get_column_index('Source_Two')).to_list()
    _z = df_data_clustered.to_series(df_data_clustered.get_column_index('Source_Three')).to_list()

    _color = df_data_clustered.to_series(df_data_clustered.get_column_index(_c.CLUSTER)).to_list()
    _inertias = df_k.to_series(df_k.get_column_index(_c.INERTIAS)).to_list()
    _k = df_k.to_series(df_k.get_column_index(_c.K)).to_list()
    _sil_score = df_k.to_series(df_k.get_column_index(_c.SIL_SCORE)).to_list()

    df_count = df_data_clustered.sort(_c.CLUSTER_NAME).group_by(_c.CLUSTER_NAME).len()
    plot_bar(x=df_count.to_series(df_count.get_column_index(_c.LEN)).to_list(), y=df_count.to_series(df_count.get_column_index(_c.CLUSTER_NAME)).to_list(), fig=fig, ax=ax3)

    plot_elbow(inertias=_inertias, fig=fig, ax=ax1)
    plot_sil(l_index=_k, sil_score=_sil_score, fig=fig, ax=ax2)

    plot_scatter(color=_color, x=_x, y=_y, filename='xy', fig=fig, ax=ax4, title='xy', xlabel='Source_One', ylabel='Source_Two')
    plot_scatter(color=_color, x=_x, y=_z, filename='xz', fig=fig, ax=ax5, title='xz', xlabel='Source_One', ylabel='Source_Three')
    plot_scatter(color=_color, x=_z, y=_y, filename='zy', fig=fig, ax=ax6, title='zy', xlabel='Source_Three', ylabel='Source_Two')

    if filename == 'analyze_sil_all':
        plot_scatter3d(x=_x, y=_y, z=_z, color=_color, filename='scatter3d')

    # fig.tight_layout()
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.90, top=0.90, wspace=0.1, hspace=0.1)

    if filename is not None and filename != '':
        fig.savefig(fname=_c.DIR_OUT+filename)

    plt.close(fig)
