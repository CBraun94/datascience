import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


# https://realpython.com/k-means-clustering-python/


_x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
_y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

DIR_OUT = r'data/output/'


def plot_elbow(inertias: list):
    r = range(1, len(inertias)+1)
    plt.plot(r, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(fname=DIR_OUT+'kmeans_elbow')
    plt.close()


def plot_scatter(kmeans, x, y):
    plt.scatter(x, y, c=kmeans.labels_)
    plt.savefig(fname=DIR_OUT+'kmeans_scatter')
    plt.close()


def get_data():
    r = None
    from . import db
    df = db.DataFrame('test')
    df.read(db.OUT)
    if df.df is not None:
        r = df.df.to_numpy()
        print(r)
    else:
        r = list(zip(_x, _y))
    return r


def elbow():
    data = get_data()
    inertias = []

    k: list[KMeans] = []

    r = range(1, len(data))

    for i in r:
        k.append(KMeans(n_clusters=i))
        k[-1].fit(data)
        inertias.append(k[-1].inertia_)

    plot_elbow(inertias=inertias)
    plot_scatter(kmeans=k[2], x=data[:, 0], y=data[:, 1])


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
