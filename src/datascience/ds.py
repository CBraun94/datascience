import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 


x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]


def plot_elbow(inertias: list):
    plt.plot(range(1, len(inertias)+1), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(fname='data/output/kmeans_elbow')
    plt.close()


def plot_scatter(kmeans):
    plt.scatter(x, y, c=kmeans.labels_)
    plt.savefig(fname='data/output/kmeans_scatter')
    plt.close()


def get_data() -> list:
    r = list(zip(x, y))
    return r


def main():
    print('main')
    data = get_data()
    inertias = []

    for i in range(1, len(data)+1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    plot_elbow(inertias=inertias)
    plot_scatter(kmeans=kmeans)


if __name__ == '__main__':
    main()
