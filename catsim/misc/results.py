"""Module with functions that take care of loading and saving CSV files with
clustering and CAT simulation results"""

import os.path
import time
import pandas
import numpy as np
from pandas import DataFrame

col_cluster = ['Data', 'Algoritmo', 'Base de dados', 'Distância', 'Variável',
               'Nº registros', 'Nº grupos', 't (segundos)', 'Menor grupo',
               'Maior grupo', 'Variância', 'Dunn', 'Silhueta',
               'Classificações', 'RMSE', 'Taxa de sobreposição']
col_cat = ['Índice', 'Data', 'Método', 't (segundos)', 'Nº de grupos',
           'Qtd. itens', 'RMSE', 'Taxa de sobreposição', 'r. max']
col_localCat = ['Índice', 'Theta', 'Est. Theta', 'Id. itens', 'r. max']


def saveClusterResults(datetime, algorithm, dataset, distance, variable,
                       n_observations, n_clusters, t, smallest_cluster,
                       largest_cluster, variance, dunn, sillhouette,
                       classifications, path):
    """Appends a result to the end of the cluster results csv file:
    """
    ar = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(datetime)),
          algorithm,
          dataset,
          distance,
          variable,
          n_observations,
          n_clusters,
          t,
          smallest_cluster,
          largest_cluster,
          variance,
          dunn,
          sillhouette,
          str(classifications.tolist()).strip('[]').replace(',', ''),
          None,
          None]

    if not os.path.exists(path):
        DataFrame([ar], columns=col_cluster).to_csv(
            path, header=True, index=False)
    with open(path, 'a') as f:
        DataFrame([ar]).to_csv(f, header=False, index=False)


def saveGlobalCATResults(index, datetime, method, t, n_clusters, qtd_itens,
                         rmse, overlap, r_max, path):
    """Appends a result to the end of the cluster results csv file:"""
    ar = [index,
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(datetime)),
          method,
          t,
          n_clusters,
          qtd_itens,
          rmse,
          overlap,
          r_max]

    if not os.path.exists(path):
        DataFrame([ar], columns=col_cat).to_csv(
            path, header=True, index=False)
    with open(path, 'a') as f:
        DataFrame([ar]).to_csv(f, header=False, index=False)


def saveLocalCATResults(index, theta, est_theta, id_itens, r_max, path):
    ar = [index,
          theta,
          est_theta,
          str(id_itens).strip('[]').replace(',', ''),
          r_max]

    if not os.path.exists(path):
        DataFrame([ar], columns=col_localCat).to_csv(
            path, header=True, index=False)
    with open(path, 'a') as f:
        DataFrame([ar]).to_csv(f, header=False, index=False)


def loadClusterResults(path):
    """Loads the csv file containing the clustering results in a
       :func: pandas.DataFrame. If the file does not exist, creates an empty
       file with the column headers
    """
    if not os.path.exists(path):
        df = pandas.DataFrame(columns=[col_cluster])
    else:
        df = pandas.read_csv(path, header=0, index_col=False,
                             encoding='utf-8')

    df['Data'] = pandas.to_datetime(df['Data'])

    df[['t (segundos)', 'Dunn', 'Silhueta', 'Nº registros', 'Nº grupos',
        'Menor grupo', 'Maior grupo', 'RMSE', 'Taxa de sobreposição']] = df[
        ['t (segundos)', 'Dunn', 'Silhueta', 'Nº registros', 'Nº grupos',
         'Menor grupo', 'Maior grupo', 'RMSE',
         'Taxa de sobreposição']].astype(np.float64)

    df[['Base de dados', 'Distância']] = df[
        ['Base de dados', 'Distância']].astype(str)

    df['Sem Classificação'] = df['Classificações'].apply(lambda x:
                                                         x.count('-1'))
    df['Classificações'] = df[
        'Classificações'].apply(
        lambda x: np.array(x.strip().strip('[]').split(' '), dtype=np.int64))

    df['pct. sem Classificação'] = df[['Sem Classificação', 'Classificações'
                                       ]].apply(lambda x: 100 / np.size(x[1]) *
                                                x[0], axis=1)

    df['Classificações'] = df['Classificações'].astype(np.ndarray)
    return df


def loadGlobalCATResults(path):
    if not os.path.exists(path):
        df = pandas.DataFrame(columns=[col_cat])
    else:
        df = pandas.read_csv(path, header=0, index_col=False,
                             encoding='utf-8')

    df[['Índice', 'Nº de grupos', 'Qtd. itens']] = df[
        ['Índice', 'Nº de grupos', 'Qtd. itens']].astype(np.int64)
    df['Data'] = pandas.to_datetime(df['Data'])
    df[['t (segundos)', 'RMSE', 'Taxa de sobreposição', 'r. max']] = df[
        ['t (segundos)', 'RMSE', 'Taxa de sobreposição', 'r. max']].astype(
            np.float64)

    return df


def loadLocalCATResults(path):
    """Loads the csv file containing the computerized adaptive testing
       simulation results in a pandas.DataFrame. If the file does not exist,
       creates an empty file with the column headers.
    """
    if not os.path.exists(path):
        df = pandas.DataFrame(columns=[col_localCat])
    else:
        df = pandas.read_csv(path, header=0, index_col=False,
                             encoding='utf-8')

    df['Índice'] = df['Índice'].astype(np.int64)
    df[['Theta', 'Est. Theta', 'r. max']] = df[
        ['Theta', 'Est. Theta', 'r. max']].astype(float)
    df['Data'] = pandas.to_datetime(df['Data'])

    df['Id. itens'] = df['Id. itens'].apply(
        lambda x: np.array(x.strip().strip('[]').split(' '), dtype=np.int64))

    return df
