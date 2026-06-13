import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


# https://realpython.com/k-means-clustering-python/


_x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
_y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

DIR_OUT = r'data/output/'

PLT_STYLE = 'dark_background'


def plot_elbow(inertias: list):
    r = range(1, len(inertias)+1)
    plt.plot(r, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(fname=DIR_OUT+'kmeans_elbow')
    plt.close()


def plot_sil(l_index: list, sil_score: list):
    plt.plot()
    plt.plot(l_index, sil_score, marker='o')
    plt.title('silhouette_score')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.xticks(l_index)
    plt.savefig(fname=DIR_OUT+'sil_score')
    plt.close()


def plot_scatter(kmeans, x, y, filename: str, title = None, xlabel = None, ylabel = None):
    plt.style.use(PLT_STYLE)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=kmeans.labels_)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    if filename is not None and filename != '':
        fig.savefig(fname=DIR_OUT+filename)


def get_data():
    _data = None
    _columns = None
    import db
    df = db.DataFrame('test')
    df.read(db.OUT)
    df.write(db.OUT)
    if df.df is not None:
        _data = df.df.to_numpy()
        _columns = df.df.columns
        print(_data)
    else:
        _data = list(zip(_x, _y))
    return _columns, _data


def elbow():
    columns, data = get_data()
    inertias = []

    k: list[KMeans] = []

    r = range(1, len(data))

    for i in r:
        k.append(KMeans(n_clusters=i))
        k[-1].fit(data)
        inertias.append(k[-1].inertia_)

    plot_elbow(inertias=inertias)
    plot_scatter(kmeans=k[7], x=data[:, 0], y=data[:, 1], filename='xy', xlabel=columns[0], ylabel=columns[1])
    plot_scatter(kmeans=k[7], x=data[:, 0], y=data[:, 2], filename='xz', xlabel=columns[0], ylabel=columns[2])
    plot_scatter(kmeans=k[7], x=data[:, 2], y=data[:, 1], filename='zy', xlabel=columns[2], ylabel=columns[1])


def sil():
    columns, data = get_data()
    inertias = []
    sil_score = []

    l_index = []
    k: list[KMeans] = []

    r = range(2, len(data))

    for i in r:
        l_index.append(i)
        k.append(KMeans(n_clusters=i))
        k[-1].fit(data)
        inertias.append(k[-1].inertia_)
        _labels = k[-1].fit_predict(data)
        sil_score.append(silhouette_score(data, _labels))

    print(sil_score)

    index = sil_score.index(max(sil_score))

    plot_elbow(inertias=inertias)
    plot_sil(l_index=l_index, sil_score=sil_score)
    plot_scatter(kmeans=k[index], x=data[:, 0], y=data[:, 1], filename='xy', xlabel=columns[0], ylabel=columns[1])
    plot_scatter(kmeans=k[index], x=data[:, 0], y=data[:, 2], filename='xz', xlabel=columns[0], ylabel=columns[2])
    plot_scatter(kmeans=k[index], x=data[:, 2], y=data[:, 1], filename='zy', xlabel=columns[2], ylabel=columns[1])


def two():
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    silhouette_coefficients = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)


def main():
    print('main')
    #elbow()
    sil()


if __name__ == '__main__':
    main()
