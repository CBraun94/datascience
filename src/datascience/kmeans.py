from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import polars as pl
import plot as p
import _const as _c


# https://realpython.com/k-means-clustering-python/


_x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
_y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]


def get_data() -> pl.DataFrame:
    r = None
    import db
    df = db.DataFrame(_c.DF_DS_NAME)
    df.read(db.OUT)
    df.write(db.OUT)
    if df.df is not None:
        r = df.df
    return r


def elbow():
    columns, data = get_data()
    inertias = []

    k: list[KMeans] = []

    r = range(1, len(data))

    for i in r:
        k.append(KMeans(n_clusters=i))
        k[-1].fit(data)
        inertias.append(k[-1].inertia_)

    p.plot_elbow(inertias=inertias)
    p.plot_scatter(kmeans=k[7], x=data[:, 0], y=data[:, 1], filename='xy', xlabel=columns[0], ylabel=columns[1])
    p.plot_scatter(kmeans=k[7], x=data[:, 0], y=data[:, 2], filename='xz', xlabel=columns[0], ylabel=columns[2])
    p.plot_scatter(kmeans=k[7], x=data[:, 2], y=data[:, 1], filename='zy', xlabel=columns[2], ylabel=columns[1])


def sil(df: pl.DataFrame = None):
    if df is None:
        df = get_data()
    columns = df.columns
    data = df.to_numpy()

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

    df_k = pl.DataFrame(data={_c.K: l_index, _c.INERTIAS: inertias, _c.SIL_SCORE: sil_score})

    if _c.DEBUG_PRINT:
        print(df_k)
        print(sil_score)

    index = sil_score.index(max(sil_score))

    s = pl.Series(name=_c.CLUSTER, values=k[index].labels_)
    sb = pl.Series(name=_c.CLUSTER_NAME, values=_c.ERROR_CLUSTER_NAME[:len(k[index].labels_)])
    df_data_clustered = df.insert_column(0, s)
    df_data_clustered = df.insert_column(1, sb)

    if _c.DEBUG_PRINT:
        print(df_data_clustered)

    analyze_sil(inertias=inertias, l_index=l_index, sil_score=sil_score, k=k, data=data, columns=columns, index=index)

    p.plot_analyze_sil(df_k, df, df_data_clustered, k[index], filename='analyze_sil_all')

    r = range(0, index)
    for i in r:
        df_to_plot = df_data_clustered.filter(pl.col(_c.CLUSTER) == i)

        p.plot_analyze_sil(df_k, df, df_to_plot, k[index], filename='analyze_sil_cluster_'+str(i))


def analyze_sil(inertias, l_index, sil_score, k, data, columns, index):
    p.plot_elbow(inertias=inertias)
    p.plot_sil(l_index=l_index, sil_score=sil_score)
    p.plot_scatter(color=k[index].labels_, x=data[:, 0], y=data[:, 1], filename='xy', xlabel=columns[0], ylabel=columns[1])
    p.plot_scatter(color=k[index].labels_, x=data[:, 0], y=data[:, 2], filename='xz', xlabel=columns[0], ylabel=columns[2])
    p.plot_scatter(color=k[index].labels_, x=data[:, 2], y=data[:, 1], filename='zy', xlabel=columns[2], ylabel=columns[1])

    df_data = {columns[0]: data[:, 0], columns[1]: data[:, 1], columns[2]: data[:, 2], _c.CLUSTER: k[index].labels_}
    df = pd.DataFrame(data=df_data).sort_values(by=[_c.CLUSTER])

    if _c.DEBUG_PRINT:
        print(pl.from_pandas(df))

    df.to_excel(_c.DIR_OUT+"output.xlsx", sheet_name=_c.SHEETNAME_OUT)

    df_m = df.groupby(_c.CLUSTER).mean()
    #df_m = df.groupby('cluster').agg(['mean', 'count'])

    if _c.DEBUG_PRINT:
        print(pl.from_pandas(df_m))

    df_m.to_excel(_c.DIR_OUT+"output_mean.xlsx", sheet_name=_c.SHEETNAME_OUT)
