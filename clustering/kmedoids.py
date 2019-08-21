from scipy.spatial.distance import squareform, pdist
import numpy as np


class KMedoids(object):
    def __init__(self, n_clusters=8, max_iter=1000, verbose=0, random_state=None, metric='euclidean', algorithm='pam'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.metric = metric
        self.algorithm = algorithm

        self.dict_clusters = None
        self.map_medoids = None
        self.labels = None
        self.inertia = None
        self.dist_x = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X):
        self.dist_x = self.fit_transform(X)

        rows, cols = self.dist_x.shape
        if self.n_clusters > cols:
            raise Exception("Error:: Too many medoids")

        if self.algorithm == 'pam':
            support_matrix, self.dict_clusters = self.kmedoids_pam(self.dist_x)
            self.map_medoids = {idx: value for idx,
                                value in enumerate(support_matrix)}
        else:
            raise Exception("Error:: Approach not implemented")

        self.labels = np.zeros(rows)

        for key in self.dict_clusters:
            for idx in self.dict_clusters[key]:
                self.labels[idx] = key

    def fit_predict(self, X):
        self.fit(X)
        self.inertia = self.compute_sse(X)
        return self.map_medoids, self.dict_clusters, self.labels, self.inertia

    def fit_transform(self, X):
        return squareform(pdist(X, metric=self.metric))

    def kmedoids_pam(self, distance_matrix):
        # randomly initialize an array of k-medoid indices
        support_matrix = np.arange(distance_matrix.shape[1])
        np.random.shuffle(support_matrix)
        support_matrix = np.sort(support_matrix[:self.n_clusters])

        # create a copy of the array of medoid indices
        new_support_matrix = np.copy(support_matrix)

        # initialize a dictionary to represent clusters
        clusters = {}

        for _ in range(self.max_iter):
            # determine clusters, i. e. arrays of data indices
            j_vector = np.argmin(distance_matrix[:, support_matrix], axis=1)
            for label in range(self.n_clusters):
                clusters[label] = np.where(j_vector == label)[0]

            # update cluster medoids
            for label in range(self.n_clusters):
                j_vector = np.mean(distance_matrix[np.ix_(
                    clusters[label], clusters[label])], axis=1)
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
            for label in range(self.n_clusters):
                clusters[label] = np.where(j_vector == label)[0]

        # return results
        return support_matrix, clusters

    def compute_sse(self, X):
        sse = 0.0
        for c in self.map_medoids:
            temp = 0.0
            for p in self.dict_clusters[c]:
                temp += self.dist_x[self.map_medoids[c], p]
                sse += temp
        return sse
