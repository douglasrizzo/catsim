import os
from time import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import catsim.cat.irt


def plot3D(points, clusters, title, centers=None):
    # plota gráficos
    # variáveis utilizadas no plot
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if title is not None:
        ax.set_title(title)

    ax.scatter(column(points.tolist(), 0), column(points.tolist(), 1),
               column(points.tolist(), 2),
               c=colors[clusters].tolist(),
               s=10)

    if centers is not None:
        center_colors = colors[:len(centers)]
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   s=100,
                   c=center_colors)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    cluster_graphdir = dissertacao_imgdir + 'clusters/'

    if not os.path.exists(cluster_graphdir):
        os.makedirs(cluster_graphdir)

    plt.savefig(cluster_graphdir + title + '.pdf', bbox_inches='tight')


def gen3DClusterGraphs():
    """Gera gráficos 3D dos itens classificados"""
    datasets = loadDatasets()
    df = loadResults()

    t0 = time()
    for counter, dataset in enumerate(datasets):
        for index, row in df[df['Dataset'] == dataset[0]].iterrows():
            print(format(counter + index + 1) +
                  ' de ' + format(len(df.index)) + '    ' +
                  format(timedelta(
                    seconds=(time.time() - t0) / (counter + index + 1) *
                    (len(df.index) - (counter + index + 1)))))

            plot3D(dataset[2], row[10], dataset[0] + ' - ' + row[0] +
                   ' (' + format(row[2]) + ')')


def genIRTGraphics():
    """
    gera curvas características e de informação dos itens para o ML3 da TRI
    """
    datasets = loadDatasets()
    total_imagens = 0
    imagem_atual = 0
    t0 = time.time()

    for dataset_name, x, x_scaled in datasets:
        total_imagens += np.size(x, 0)

    for dataset_name, x, x_scaled in datasets:

        print('\nGerando gráficos, base: ' + dataset_name)
        for counter, triple in enumerate(x):
            imagem_atual += 1
            print(format(imagem_atual),
                  'de', format(total_imagens),
                  '  ',
                  format(timedelta(seconds=((time.time() - t0) /
                                            imagem_atual) *
                                           (total_imagens - imagem_atual))),
                  '\r',
                  end='\r')
            p_thetas = []
            i_thetas = []
            thetas = np.arange(triple[1] - 4, triple[1] + 4, .1, 'double')

            for theta in thetas:
                p_thetas.append(irt.tpm(theta, triple[0],
                                        triple[1], triple[2]))
                i_thetas.append(irt.inf(theta, triple[0],
                                        triple[1], triple[2]))

            tri_graphdir = '/home/douglas/Desktop/teste/'

            if not os.path.exists(tri_graphdir):
                os.makedirs(tri_graphdir)

            plt.figure()
            plt.title(dataset_name + ' - ' + format(counter + 1), size=18)
            plt.annotate('$a = ' + format(triple[0]) + '$\n$b = ' + format(
                triple[1]) + '$\n$c = ' + format(triple[2]) + '$',
                bbox=dict(facecolor='white',
                          alpha=1),
                xy=(.75, .05),
                xycoords='axes fraction')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$P(\theta)$')
            plt.grid()
            plt.legend(loc='best')
            plt.plot(thetas, p_thetas)
            plt.savefig(tri_graphdir + '/' + dataset_name + '_' +
                        format(counter + 1) + '_prob.pdf')

            plt.figure()
            plt.title(dataset_name + ' - ' + format(counter + 1), size=18)
            plt.annotate('$a = ' + format(triple[0]) + '$\n$b = ' + format(
                triple[1]) + '$\n$c = ' + format(triple[2]) + '$',
                bbox=dict(facecolor='white',
                          alpha=1),
                xy=(.75, .05),
                xycoords='axes fraction')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$I(\theta)$')
            plt.grid()
            plt.legend(loc='best')
            plt.plot(thetas, i_thetas)
            plt.savefig(tri_graphdir + '/' + dataset_name + '_' +
                        format(counter + 1) + '_info.pdf')

    print('Término impressão gráficos TRI, ' + format(total_imagens) +
          ' imagens\nTempo: ' + format(timedelta(seconds=time.time() - t0)))
