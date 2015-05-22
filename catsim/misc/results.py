import os
import pandas
from pandas import DataFrame
import numpy as np

columns = ['Data', 'Algoritmo', 'Dataset', 'Variável', 'Nº registros',
           'Nº grupos', 't (segundos)', 'Menor cluster', 'Maior cluster',
           'Variância', 'Dunn', 'Silhueta', 'Classificações']


def loadResults(path):
    """
    Caso o arquivo cluster_results.csv já existe, ele pode ser carregado usando
    esta função
    """
    if not os.path.exists(path):
        df = pandas.DataFrame([columns])
    else:
        df = pandas.read_csv(path, header=0, encoding='latin_1')

    df[['Data', 't (segundos)', 'Dunn', 'Silhueta', 'Nº registros', 'Variável',
        'Nº grupos', 'Menor cluster', 'Maior cluster']] = df[
        ['t (segundos)', 'Dunn', 'Silhueta', 'Nº registros',
         'Variável', 'Nº grupos', 'Menor cluster',
         'Maior cluster']].astype(float)

    df['Sem classificação'] = df['Classificações'].apply(lambda x:
                                                         x.count('-1'))
    df['Classificações'] = df['Classificações'].apply(lambda x:
                                                      np.array(x.split(' '),
                                                               dtype='int'))
    df['pct. sem classificação'] = df[['Sem classificação', 'Classificações'
                                       ]].apply(lambda x: 100 / np.size(x[1]) *
                                                x[0], axis=1)
    return df


def saveResults(df, path):
    '''
    Appends a result to the end of the CSV file:

    input : a pandas.DataFrame in which each column corresponds to the
            following:

    datetime, clustering_time, dunn, sillhouette, n_itens,
    algorithm_specific_variable, n_clusters, smallest_cluster, largest_cluster,
    group_indexes
    '''
    if not os.path.exists(path):
        DataFrame([df], columns=columns).to_csv(path, header=True, index=False)
    with open(path, 'a') as f:
        DataFrame([df]).to_csv(f, header=False, index=False)


def process(datadir, imgdir):
    df = loadResults()

    df.groupby('Algoritmo')['Menor cluster', 'Maior cluster', 't (segundos)',
                            'Variância', 'Dunn', 'Silhueta'].mean().to_csv(
                                datadir + 'alg_means.csv')
    df.groupby('Dataset')['Menor cluster', 'Maior cluster', 't (segundos)',
                          'Variância', 'Dunn', 'Silhueta'].mean().to_csv(
        datadir + 'dataset_means.csv')
    df.groupby('Nº grupos')['Menor cluster', 'Maior cluster', 't (segundos)',
                            'Variância', 'Dunn', 'Silhueta'].mean().to_csv(
                                datadir + 'nclusters_means.csv')

    ax = df.groupby(
        'Nº grupos')['Variância', 'Dunn', 'Silhueta'].mean().plot(
            title='Índices de validação de clusters / Nº grupos',
            legend='best',
            figsize=(8, 6))
    ax.set_ylabel('Índices')
    ax.get_figure().savefig(imgdir + 'validity_by_nclusters.pdf')

    df_enem = df[df['Dataset'] == 'Enem'][df['Algoritmo'] !=
                                          'Aff. Propagation'][df['Algoritmo']
                                                              != 'DBSCAN']
    df_sintetico = df[df['Dataset'] ==
                      'Sintético'][df['Algoritmo'] !=
                                   'Aff. Propagation'][df['Algoritmo'] !=
                                                       'DBSCAN']

    ax = pandas.pivot_table(
        df_enem,
        values='Dunn',
        columns='Algoritmo',
        index='Nº grupos').plot(
            figsize=(8, 6),
            grid=True,
            title='Média Dunn / Algoritmo na base \'Enem\'')
    ax.set_ylabel('Dunn')
    ax.get_figure().savefig(imgdir + 'dunn_by_algorithm_enem.pdf')

    ax = pandas.pivot_table(
        df_enem,
        values='Silhueta',
        columns='Algoritmo',
        index='Nº grupos').plot(
            figsize=(8, 6),
            grid=True,
            title='Média silhueta / Algoritmo na base \'Enem\'')
    ax.set_ylabel('Silhueta')
    ax.get_figure().savefig(
        imgdir + 'silhouette_by_algorithm_enem.pdf')

    ax = pandas.pivot_table(
        df_enem,
        values='Menor cluster',
        columns='Algoritmo',
        index='Nº grupos').plot(
            figsize=(8, 6),
            grid=True,
            title='Itens no menor cluster / Algoritmo na base \'Enem\'')
    ax.set_ylabel('Itens no menor cluster')
    ax.get_figure().savefig(
        imgdir + 'smallestcluster_by_algorithm_enem.pdf')

    ax = pandas.pivot_table(df_sintetico, values='Dunn',
                            columns='Algoritmo', index='Nº grupos').plot(
        figsize=(8, 6),
        grid=True,
        title='Média Dunn /' +
        ' Algoritmo na base \'Sintética\'')
    ax.set_ylabel('Dunn')
    ax.get_figure().savefig(imgdir +
                            'dunn_by_algorithm_sintetico.pdf')

    ax = pandas.pivot_table(df_sintetico, values='Silhueta',
                            columns='Algoritmo', index='Nº grupos').plot(
        figsize=(8, 6), grid=True, title='Média silhueta' +
        '/ Algoritmo na base \'Sintética\'')
    ax.set_ylabel('Silhueta')
    ax.get_figure().savefig(
        imgdir + 'silhouette_by_algorithm_sintetico.pdf')

    ax = pandas.pivot_table(df_sintetico, values='Menor cluster',
                            columns='Algoritmo', index='Nº grupos').plot(
        figsize=(8, 6), grid=True, title='Itens no menor' +
        'cluster / Algoritmo na base \'Sintética\'')
    ax.set_ylabel('Itens no menor cluster')
    ax.get_figure().savefig(
        imgdir + 'smallestcluster_by_algorithm_sintetico.pdf')

    dfdb = df[df['Algoritmo'] == 'DBSCAN']
    dfdb = dfdb.rename(columns={'Variável': '$\epsilon$'})

    ax = pandas.pivot_table(dfdb,
                            values='Dunn',
                            columns='Dataset',
                            index='$\epsilon$').plot(
                                figsize=(8, 6),
                                grid=True,
                                title='Média Dunn / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Dunn')
    ax.get_figure().savefig(imgdir + 'dunn_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='Silhueta',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='Média silhueta / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Silhueta')
    ax.get_figure().savefig(imgdir + 'silhouette_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='Menor cluster',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='Itens no menor cluster / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Itens no menor cluster')
    ax.get_figure().savefig(
        imgdir + 'smallestcluster_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='pct. sem classificação',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='% de itens não classificados / $\epsilon$ para DBSCAN')
    ax.set_ylabel('% Itens')
    ax.get_figure().savefig(imgdir + 'unclassified_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='Nº grupos',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='Nº de clusters / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Nº grupos')
    ax.get_figure().savefig(imgdir + 'nclusters_by_dbscan.pdf')
