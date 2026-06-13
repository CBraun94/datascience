import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

    ccc = []
    r = range(0, len(inertias)-1)
    for i in r:
        aaa = (inertias[i+1]/inertias[i])
        ccc.append(aaa)
        print(str(i) + ' ' + str(i+1) + ' ' + str(aaa))


    ddd = []
    r = range(1, len(ccc)-1)
    for i in r:
        bbb = ccc[i-1] + ccc[i] + ccc[i+1]
        bbb = bbb / 3
        ddd.append(bbb)

        print(str(i-1) + ' ' + str(i) + ' ' + str(i+1) + ' ' + str(bbb))

    print('aaaaaaa')
    print(max(ddd))

    plot_elbow(inertias=inertias)
    plot_scatter(kmeans=k[7], x=data[:, 0], y=data[:, 1], filename='xy', xlabel=columns[0], ylabel=columns[1])
    plot_scatter(kmeans=k[7], x=data[:, 0], y=data[:, 2], filename='xz', xlabel=columns[0], ylabel=columns[2])
    plot_scatter(kmeans=k[7], x=data[:, 2], y=data[:, 1], filename='zy', xlabel=columns[2], ylabel=columns[1])


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
    elbow()


if __name__ == '__main__':
    main()
