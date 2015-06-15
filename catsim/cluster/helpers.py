def normalize(c):
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
