#!/Users/alyssavance/anaconda3/bin/python

import csv
import math
import numpy as np
import pickle
import sqlite3

from scipy.sparse import coo_matrix
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import scale
from sklearn.utils import check_array
from sklearn.utils.graph import graph_shortest_path

sql_database = '/Users/alyssavance/hackathon/metaTables.sqlite'
csv_table = "/Users/alyssavance/hackathon/test_table.csv"
id_map_out = "/Users/alyssavance/hackathon/person_id_map.csv"
pickle_2d_out = "/Users/alyssavance/hackathon/large_clusters.pickle"

output_dim_count = 2
max_clusters = 2
dim_downsample = 1
gmm_downsample = 1

cost_col_list = {0, 1, 21, 29, 31, 33, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51}
enum_col_list = {2, 3, 4, 9, 22, 23, 25, 28, 36, 37, 38, 39, 40, 42}
fold_col_list = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
ignore_col_list = {6, 35}
nosum_col_list = {7, 24}
person_id_col = 35
fold_col = 9

conn = sqlite3.connect(sql_database)
c = conn.cursor()
np.set_printoptions(precision=3)

# Monkey patching around sklearn bug
### BEGIN MONKEYPATCHING
def _fit_transform(self, X):
    X = check_array(X, accept_sparse=True)
    self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                  algorithm=self.neighbors_algorithm,
                                  n_jobs=self.n_jobs)
    self.nbrs_.fit(X)
    self.training_data_ = self.nbrs_._fit_X
    self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                 kernel="precomputed",
                                 eigen_solver=self.eigen_solver,
                                 tol=self.tol, max_iter=self.max_iter,
                                 n_jobs=self.n_jobs)

    kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                           mode='distance', n_jobs=self.n_jobs)

    self.dist_matrix_ = graph_shortest_path(kng,
                                            method=self.path_method,
                                            directed=False)
    G = self.dist_matrix_ ** 2
    G *= -0.5

    self.embedding_ = self.kernel_pca_.fit_transform(G)

Isomap._fit_transform = _fit_transform
### END MONKEYPATCHING

def csv_reader(filename):
    with open(filename, 'r') as csvfile:
        rowlist = []
        for row in csv.reader(csvfile):
            rowlist.append(row)

        return rowlist

def fetch_all(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()

def fetch_one(cursor, query):
    cursor.execute(query)
    return cursor.fetch_one()

def print_col_info():
    col_defs = fetch_all(c, "PRAGMA table_info('metaClaims');")
    col_vals = fetch_all(c, "SELECT * from metaClaims LIMIT 10 OFFSET 27;")
    for col in list(zip(col_defs, zip(*col_vals))):
        print(col)

def build_sparse_matrix(points):
    data, row, col = list(zip(*points))
    return coo_matrix((data, (row, col)))

def count_col(c):
    cset = set(c)
    return dict(zip(cset, range(len(cset))))

def generate_col_maps(table):
    col_maps = {}
    for i in enum_col_list:
        col_maps[i] = count_col(table[i])

    fold_list = []
    fold_cols = fold_col_list | {fold_col}
    for i in fold_cols:
        fold_list.extend(table[i])
        
    fold_map = count_col(fold_list)
    for i in fold_cols:
        col_maps[i] = fold_map

    return col_maps

def val_append(triple_dict, val, rownum, colnum):
    k = (rownum, colnum)
    if k in triple_dict:
        triple_dict[k] += val
    else:
        triple_dict[k] = val

def log_val_append(triple_dict, val, rownum, colnum):
    k = (rownum, colnum)
    if k in triple_dict:
        try:
            triple_dict[k] = math.log(abs(val) + math.exp(triple_dict[k]))
        except OverflowError:
            print("overflow error")
            print(val)
            print(triple_dict[k])
            print(rownum)
            print(colnum)
    else:
        triple_dict[k] = math.log(abs(val))
        
def process_row(triple_dict, col_maps, rownum, row):
    col_count = 0

    for i in range(0, len(row)):
        if i in enum_col_list:
            col_val = col_count + col_maps[i][row[i]]
            val_append(triple_dict, 1.0, rownum, col_val)
            col_count += len(col_maps[i])
        elif i in fold_col_list:
            col_val = col_count + col_maps[i][row[i]]
            val_append(triple_dict, 1.0, rownum, col_val)
        elif i in cost_col_list:
            if row[i] != "" and float(row[i]) != 0.0:
                val_append(triple_dict, float(row[i]), rownum, col_count)
            col_count += 1
        elif i in nosum_col_list:
            if row[i] != "" and float(row[i]) != 0.0:
                triple_dict[(rownum, col_count)] = float(row[i])
            col_count += 1
        elif i in ignore_col_list:
            pass
        else:
            try:
                if row[i] != "" and float(row[i]) != 0.0:
                    val_append(triple_dict, float(row[i]), rownum, col_count)
            except ValueError:
                pass
            col_count += 1
            
def process_rows(csv_table):
    print("Splitting table into columns")
    zip_table = list(zip(*csv_table))
    print("Generating column maps")
    col_maps = generate_col_maps(zip_table)
    person_id_map = count_col(zip_table[person_id_col])

    with open(id_map_out, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for k in person_id_map:
            writer.writerow([k, person_id_map[k]])
    
    triple_dict = {}
    row_counter = 0

    print("Processing rows")
    for row in csv_table:
        row_counter += 1
        process_row(triple_dict, col_maps, person_id_map[row[person_id_col]], row)
        
        if row_counter % 10000 == 0:
            print("Processed row " + str(row_counter))

    print("Assembling triple list")
    triples = []
    for k in triple_dict:
        triples.append([triple_dict[k], k[0], k[1]])

    return triples

def print_dim_variance(svd_model):
    print("Variance explained per dimension:")
    for x in svd_model.explained_variance_ratio_:
        print(x)
    print("Total:")
    print(sum(svd_model.explained_variance_ratio_))
    
def fit_svd(matrix):
    svd = TruncatedSVD(n_components=output_dim_count)
    svd_model = svd.fit(matrix)
    print_dim_variance(svd_model)
    return svd_model.transform(matrix)

def fit_isomap(matrix):
    isomap = Isomap(n_components=output_dim_count)
    isomap_model = isomap.fit(matrix)
    print("Reconstruction error")
    print(isomap_model.reconstruction_error())
    return isomap.fit_transform(matrix)

def fit_lle(matrix):
    lle = LocallyLinearEmbedding(n_components=output_dim_count)
    lle_model = lle.fit(matrix)
    return lle_model.transform(matrix)

def fit_tsne(matrix):
    tsne = TSNE(n_components=3)
    return tsne.fit_transform(matrix)

def bayesian_gmm(matrix):
    vbgmm = BayesianGaussianMixture(n_components=max_clusters,
                                    covariance_type='full',
                                    verbose=2, tol=1e-5,
                                    verbose_interval=1)
    return vbgmm.fit(matrix)

def log_transform(r):
    if r < 0:
        return -1 * math.log(-1 * r)
    elif r == 0:
        return 0
    else:
        return math.log(r)

def truncate_data(matrix):
    data_2d = []
    for r in matrix:
        data_2d.append(r[0:3])
    return data_2d
    
def pickle_out(matrix):
    pickle.dump(matrix, open(pickle_2d_out, "wb"))    

print("Starting program")
csv_data = csv_reader(csv_table)
print("Loaded data from CSV")
triples = process_rows(csv_data)
print("Building matrix")
matrix = build_sparse_matrix(triples)
#print("Number of people is:")
#print(matrix.getnnz())
print("Downsampling")
down_matrix = matrix.tolil()[::dim_downsample, ::1]
print("Scaling matrix")
scaled_matrix = scale(down_matrix.tocsc(), with_mean=False)
print("Starting dimensionality reduction")
reduced_data = fit_isomap(scaled_matrix)
print("Reduction finished")
vbgmm = bayesian_gmm(reduced_data[::gmm_downsample,:])
print("Computed Gaussian mixture model")
print("Distribution of cluster weights:")
w = vbgmm.weights_
w.sort()
print(w)
truncated_data = fit_tsne(reduced_data)
labels = vbgmm.predict(reduced_data)
pickle_out(list(zip(truncated_data, labels)))
print("Finished pickling")
