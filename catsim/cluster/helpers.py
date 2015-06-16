def normalize(c):
    """Normalize the clustering results list, so that they are comparable
       between each other.

    :param c: a vector containing cluster memberships
    :type c: numpy.ndarray
    :return: a vector containing normalized cluster memberships
    :rtype: numpy.ndarray

    :note: as an example of what is meant by 'normalization', imagine the
           following cluster membership vectors:

            .. code-block:: python

               [2, 2, 3, 2, 1, 0]
               [1, 1, 2, 1, 3, 0]
           
           In this case, both membership vectors can be considered equal, since
           they cluster data points in equal clusters. What this function does
           is transform both of them in the following:
           
            .. code-block:: python
           
               [0, 0, 1, 0, 2, 3]

           In this way, membership vectors are comparable.
    """
    c += max(c)  # shift all assginments away from their original values
    bogus = max(c) + 1  # get a bogus value that is not in the original array

    # initialize an array with the bogus value and fill it with the cluster
    # assignments in the order they appear
    natural_assignments = [bogus] * len(set(c))
    for i in range(len(natural_assignments)):
        for ii in c:
            if ii not in natural_assignments:
                natural_assignments[i] = ii
                break

    # clusters are assigned in the order they appear, eg. the first cluster is
    # 0, the second is 1 and so on. apprently, it skips -1, which is 'no
    # assignment', so this characteristic is preserved
    for i in range(len(natural_assignments)):
        c[c == i] = bogus
        c[c == natural_assignments[i]] = i

    return c
