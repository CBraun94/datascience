import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 


x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

DIR_OUT = r'data/output/'


def plot_elbow(inertias: list):
    r = range(1, len(inertias)+1)
    plt.plot(r, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(fname=DIR_OUT+'kmeans_elbow')
    plt.close()


def plot_scatter(kmeans):
    plt.scatter(x, y, c=kmeans.labels_)
    plt.savefig(fname=DIR_OUT+'kmeans_scatter')
    plt.close()


def get_data() -> list:
    r = list(zip(x, y))
    return r


def main():
    print('main')
    data = get_data()
    inertias = []

    k: list[KMeans] = []

    r = range(1, len(data))

    for i in r:
        k.append(KMeans(n_clusters=i))
        k[-1].fit(data)
        inertias.append(k[-1].inertia_)

    plot_elbow(inertias=inertias)
    plot_scatter(kmeans=k[1])


if __name__ == '__main__':
    main()
