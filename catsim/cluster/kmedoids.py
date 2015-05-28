import numpy as np


def kmedoids(D, k, iters=100):

    # determine dimensions of distance matrix D
    m, n = D.shape
    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))
    # create a copy of the array of medoid indices
    Mnew = np.copy(M)
    # initialize a dictionary to represent clusters
    C = {}

    for t in range(iters):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)

        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]

        np.sort(Mnew)

        # check for convergence

        if np.array_equal(M, Mnew):
            break

        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C


def kmedoids_to_sklearn(C):
    Cnew = np.zeros(sum(len(a) for a in C.values()) + 1)

    for c, cis in C.items():
        for ci in cis:
            Cnew[ci] = c

    # return results
    return Cnew
