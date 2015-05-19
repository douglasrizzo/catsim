import os
import numpy as np
import matplotlib.pyplot as plt

def plot3D(points, clusters, title, centers=None):
    # plota gráficos
    # variáveis utilizadas no plot
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if title != None:
        #plt.title(dataset_name +' - '+algorithm_name, size=18)
        ax.set_title(title)

    ax.scatter(column(points.tolist(), 0), column(points.tolist(), 1),
               column(points.tolist(), 2),
               c=colors[clusters].tolist(),
               s=10)

    if centers != None:
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