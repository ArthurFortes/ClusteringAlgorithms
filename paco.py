"""

PaCo Algorithm
Co-Clustering Algorithm

Paper: Improving Co-Cluster Quality with Application to Product Recommendations
Doi: http://dl.acm.org/citation.cfm?id=2661980
Authors: Michail Vlachos, Francesco Fusco, Charalambos Mavroforakis, Anastasios Kyrillidis, a
nd Vassilios G. Vassiliadis.
2014

"""

from caserec.utils.read_file import ReadFile
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
import numpy as np
import itertools

__author__ = 'Arthur Fortes'


class PaCo(object):
    def __init__(self, training_file, k_row=5, k_col=5, density_low=0.008):
        """
        :param training_file: (string:: file)
        :param k_row: (int) number of clusters generated by k-means in rows
        :param k_col: (int) number of clusters generated by k-means in rows
        :param density_low: (float) threshold to change the density matrix values
        """

        self.training_set = ReadFile(training_file).return_information(implicit=True)
        self.k_row = k_row
        self.k_col = k_col
        self.density_low = density_low

        self.list_row = [list() for _ in range(self.k_row)]
        self.list_col = [list() for _ in range(self.k_col)]

        self.count_total, self.count_ones = list(), list()
        self.density = None
        self.delta_entropy = list()

    def run_kmeans(self):
        # Call kmeans to rows and cols
        clusters_rows = KMeans(n_clusters=self.k_row).fit(self.training_set['matrix'])
        clusters_cols = KMeans(n_clusters=self.k_col).fit(self.training_set['matrix'].T)

        # Map inverse index
        [self.list_row[label].append(row_id) for row_id, label in enumerate(clusters_rows.labels_)]
        [self.list_col[label].append(col_id) for col_id, label in enumerate(clusters_cols.labels_)]

    def count_information(self):
        for label_row in range(self.k_row):
            for label_col in range(self.k_col):
                count_local = 0

                for pair in itertools.product(self.list_row[label_row], self.list_col[label_col]):
                    if self.training_set['matrix'][pair[0]][pair[1]] != 0:
                        count_local += 1

                self.count_total.append(len(self.list_row[label_row]) * len(self.list_col[label_col]))
                self.count_ones.append(count_local)

        self.update_information(first_iteration=True)

    def update_information(self, first_iteration=False):
        """
        :param first_iteration: (bool) if True calculate self.count_total and self.count_ones
        """

        if first_iteration:
            self.count_total = np.matrix(self.count_total).reshape((self.k_row, self.k_col))
            self.count_ones = np.matrix(self.count_ones).reshape((self.k_row, self.k_col))
            self.density = np.matrix(np.divide(self.count_ones, self.count_total))
            # self.density = np.matrix(np.divide(self.count_ones, self.count_total)).reshape((self.k_row, self.k_col))
        else:
            self.density = np.matrix(np.divide(self.count_ones, self.count_total))
            self.density[self.density < self.density_low] = .0

    def calculate_entropy(self):
        total_density = self.density.sum()
        probability = np.divide(self.density, total_density)

        sum_pi = 0
        for pi in probability.flat:
            sum_pi += 0 if pi == 0 else pi * np.log2(pi)

        return (-sum_pi) / np.log2(probability.size)

    @staticmethod
    def return_min_value(matrix):
        min_value = (float('inf'), (0, 0))
        for i in range(len(matrix)):
            for j in range(i):
                if matrix[i][j] < min_value[0]:
                    min_value = (matrix[i][j], (i, j))

        return min_value

    def merge(self, min_value_row, min_value_col):

        if min_value_row[0] > min_value_col[0]:

            # merge of columns
            pair = min_value_col[1]

            new_set_col = self.list_col[pair[0]].copy() + self.list_col[pair[1]].copy()
            self.list_col = list(np.delete(self.list_col, [pair[0], pair[1]], axis=0))
            self.list_col.append(new_set_col)

            # update count total based on columns
            new_count_total = self.count_total[:, pair[0]] + self.count_total[:, pair[1]]
            self.count_total = np.delete(self.count_total, (pair[0], pair[1]), axis=1)
            self.count_total = np.insert(self.count_total, self.count_total.shape[1], new_count_total.T, axis=1)

            # update count ones based on columns
            new_count_ones = self.count_ones[:, pair[0]] + self.count_ones[:, pair[1]]
            self.count_ones = np.delete(self.count_ones, (pair[0], pair[1]), axis=1)
            self.count_ones = np.insert(self.count_ones, self.count_ones.shape[1], new_count_ones.T, axis=1)

        else:
            # merge of rows
            pair = min_value_row[1]

            new_set_row = self.list_row[pair[0]].copy() + self.list_row[pair[1]].copy()
            self.list_row = list(np.delete(self.list_row, [pair[0], pair[1]], axis=0))
            self.list_row.append(new_set_row)

            # update count total based on rows
            new_count_total = self.count_total[pair[0], :] + self.count_total[pair[1], :]
            self.count_total = np.delete(self.count_total, (pair[0], pair[1]), axis=0)
            self.count_total = np.insert(self.count_total, self.count_total.shape[0], new_count_total, axis=0)

            # update count ones based on rows
            new_count_ones = self.count_ones[pair[0], :] + self.count_ones[pair[1], :]
            self.count_ones = np.delete(self.count_ones, (pair[0], pair[1]), axis=0)
            self.count_ones = np.insert(self.count_ones, self.count_ones.shape[0], new_count_ones, axis=0)

        self.update_information()

    def train_model(self):
        count_epoch = 0
        criteria = True
        # 1st step: run k-means
        self.run_kmeans()
        # 2st step: collect information (only one time)
        self.count_information()

        entropy0 = self.calculate_entropy()

        # 3st step: training the algorithm
        while criteria:
            old_density, old_list_row, old_list_col = self.density.copy(), self.list_row.copy(), self.list_col.copy()
            distance_rows = np.divide(np.float32(squareform(pdist(self.density, 'euclidean'))), self.density.shape[1])
            distance_cols = np.divide(np.float32(squareform(pdist(self.density.T, 'euclidean'))), self.density.shape[0])
            min_row = self.return_min_value(distance_rows)
            min_col = self.return_min_value(distance_cols)

            self.merge(min_row, min_col)

            # Check the number os bi-clusters
            if len(self.list_row) == 1 and len(self.list_col) == 1:
                break

            entropy = self.calculate_entropy()
            dif_entropy = entropy - entropy0
            self.delta_entropy.append(dif_entropy)
            mean_range, std_range = np.mean(self.delta_entropy), np.std(self.delta_entropy)

            if not (mean_range - 3 * std_range <= dif_entropy <= mean_range + 3 * std_range):
                self.density, self.list_row, self.list_col = old_density, old_list_row, old_list_col
                criteria = False
            else:
                print('Epoch:: ', count_epoch, " | Entropy:: ", entropy)
                entropy0 = entropy
                count_epoch += 1

        return entropy0

    def return_bi_groups(self):
        print(self.density[np.logical_and(self.density != 1, self.density != 0)])

	# filters the bigroups removing the ones with lower density, leaving the minimum to recommend to every user
    def filter_bi_groups(self):
        filteredDensities = self.density.copy()
        filteredDensities[filteredDensities == 1] = np.nan

        firstRun = True
        old = filteredDensities.copy()
        while (True):
            for line in filteredDensities:
                if np.nansum(line) == 0:
                    if firstRun:
                        return not np.isnan(self.density)
                    else:
                        return not np.isnan(old)

            old = filteredDensities.copy()

            index = np.argmin(filteredDensities)
            line = index // filteredDensities.shape[1]
            col = index % filteredDensities.shape[1]

            filteredDensities[line, col] = np.nan

            firstRun = False
		
    def execute(self):
        print("Final entropy::", self.train_model())
        print("Number of rows:: ", len(self.list_row), "Number of columns:: ", len(self.list_col))
        print("Number of bi-groups:: ", len(self.list_row) * len(self.list_col))
        print("Number of bi-groups needing recommendations:: ", self.density[np.logical_and(self.density != 1,
                                                                                            self.density != 0)].size)
        self.return_bi_groups()