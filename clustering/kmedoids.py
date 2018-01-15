import numpy as np


def kmedoids(distance_matrix, k, max_interactions=10000):
    # determine dimensions of distance matrix
    row, col = distance_matrix.shape

    if k > col:
        raise Exception("Error:: Too many medoids")

    # randomly initialize an array of k-medoid indices
    support_matrix = np.arange(col)
    np.random.shuffle(support_matrix)
    support_matrix = np.sort(support_matrix[:k])

    # create a copy of the array of medoid indices
    new_support_matrix = np.copy(support_matrix)

    # initialize a dictionary to represent clusters
    clusters = {}

    for _ in range(max_interactions):
        # determine clusters, i. e. arrays of data indices
        j_vector = np.argmin(distance_matrix[:, support_matrix], axis=1)
        for label in range(k):
            clusters[label] = np.where(j_vector == label)[0]

        # update cluster medoids
        for label in range(k):
            j_vector = np.mean(distance_matrix[np.ix_(clusters[label], clusters[label])], axis=1)
            j = np.argmin(j_vector)
            new_support_matrix[label] = clusters[label][j]
        np.sort(new_support_matrix)

        # check for convergence
        if np.array_equal(support_matrix, new_support_matrix):
            break
        support_matrix = np.copy(new_support_matrix)

    else:
        # final update of cluster memberships
        j_vector = np.argmin(distance_matrix[:, support_matrix], axis=1)
        for label in range(k):
            clusters[label] = np.where(j_vector == label)[0]

    # return results
    return support_matrix, clusters
