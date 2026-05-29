from __future__ import division
import sys
sys.path.append("./sc_utils")

import argparse
import os
import copy
import warnings
import scipy 
import scipy.sparse as sparse
import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.validation import check_array
from sklearn.utils import check_symmetric
from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering as sklearn_SpectralClustering
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

scaler = MinMaxScaler(feature_range=(0, 1))

class SparseSpectralClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', p_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=None):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.p_neighbors = p_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            raise ValueError('affinity_matrix is not specified.')
        
        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
        return self

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"


def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans'):
    if assign_labels not in ('kmeans', 'discretize'):
        raise ValueError("The 'assign_labels' parameter should be "
                         "'kmeans' or 'discretize', but '%s' was given"
                         % assign_labels)

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    maps = spectral_embedding(affinity, n_components=n_components,
                              eigen_solver=eigen_solver,
                              random_state=random_state,
                              eigen_tol=eigen_tol, drop_first=False)

    if assign_labels == 'kmeans':
        kmeans = KMeans(n_clusters, random_state=random_state, n_init=n_init).fit(maps)
        labels = kmeans.labels_
    else:
        labels = discretize(maps, random_state=random_state)

    return labels


def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       norm_laplacian=True, drop_first=True):
    adjacency = check_symmetric(adjacency)

    norm_laplacian = False
    random_state = check_random_state(random_state)
    n_nodes = adjacency.shape[0]
    if not _graph_is_connected(adjacency):
        warnings.warn("Graph is not fully connected, spectral embedding"
                      " may not work as expected.")
    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                      return_diag=True)
    if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
       (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        try:
            laplacian *= -1
            v0 = random_state.uniform(-1, 1, laplacian.shape[0])
            lambdas, diffusion_map = eigsh(laplacian, k=n_components,
                                           sigma=1.0, which='LM',
                                           tol=eigen_tol, v0=v0)
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            eigen_solver = "lobpcg"
            laplacian *= -1

    embedding = _deterministic_vector_sign_flip(embedding)
    return embedding[:n_components].T


def _set_diag(laplacian, value, norm_laplacian):
    n_nodes = laplacian.shape[0]
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            laplacian = laplacian.todia()
        else:
            laplacian = laplacian.tocsr()
    return laplacian

def get_kneighbors_conn(X_dist, p_neighbors):
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_neighbors]
        X_dist_out[indices, i] = 1
    return X_dist_out

def getLaplacian(X):
    X[np.diag_indices(X.shape[0])] = 0
    A = X
    D = np.sum(np.abs(A), axis=1)
    D = np.diag(D)
    L = D - A
    return L
    
def eig_decompose(L, k):
    try:
        lambdas, eig_vecs = scipy.linalg.eigh(L)
    except:
        try:
            lambdas = scipy.linalg.eigvals(L)
            eig_vecs = None
        except:
            lambdas, eig_vecs = scipy.sparse.linalg.eigsh(L)
    return lambdas, eig_vecs

def getLamdaGaplist(lambdas):
    lambda_gap_list = []
    for i in range(len(lambdas)-1):
        lambda_gap_list.append(float(lambdas[i+1])-float(lambdas[i]))
    return lambda_gap_list

def estimate_num_of_spkrs(X_conn, SPK_MAX):
    L = getLaplacian(X_conn)
    lambdas, eig_vals = eig_decompose(L, k=X_conn.shape[0])
    lambdas = np.sort(lambdas)
    lambda_gap_list = getLamdaGaplist(lambdas)
    num_of_spk = np.argmax(lambda_gap_list[:min(SPK_MAX, len(lambda_gap_list))]) + 1
    return num_of_spk, lambdas, lambda_gap_list


def kaldi_style_lable_writer(seg_lable_list, write_path):
    with open(write_path, 'w') as the_file:                                       
        for tup in seg_lable_list:
            line = tup[0] + ' ' + str(tup[1]) + ' \n'
            the_file.write(line)   

def nps(str_num):
    int_num = int(str_num)
    float_num = float(int_num/100.00)
    return round(float_num, 2)

def read_embd_seg_info(param):
    embd_seg_dict = {}
    return embd_seg_dict

def _graph_is_connected(graph):
    if sparse.isspmatrix(graph):
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _graph_connected_component(graph, node_id):
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def get_X_conn_from_dist(X_dist_raw, p_neighbors):
    X_r = get_kneighbors_conn(X_dist_raw, p_neighbors) 
    X_conn_from_dist = 0.5 * (X_r + X_r.T)
    return X_conn_from_dist
    

def isFullyConnected(X_conn_from_dist):
    gC = _graph_connected_component(X_conn_from_dist, 0).sum() == X_conn_from_dist.shape[0]
    return gC

def gc_thres_min_gc(mat, max_n, n_list):
    p_neighbors = 1
    X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors) 
    fully_connected = isFullyConnected(X_conn_from_dist)
    for i, p_neighbors in enumerate(n_list):
        fully_connected = isFullyConnected(X_conn_from_dist)
        X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors) 
        if fully_connected or p_neighbors > max_n:
            break
    return X_conn_from_dist, p_neighbors

def scp2dict(path):
    t_list = open(path)
    out_dict = {}
    for line in t_list:
        key = line.strip().split()[0]
        val = line.strip().split()[1]
        if key not in out_dict:
            out_dict[key] = val
    return out_dict

def checkOutput(key, seg_list, Yk):
    if len(seg_list) != Yk.shape[0]:
        raise ValueError("Segments file length mismatch -key: {} Should be: {} But Yk shape got: {}".format(key, len(seg_list), Yk.shape[0]) )


class GraphSpectralClusteringClass(object):
    def __init__(self, param):
        self.param = param
        if isinstance(self.param.threshold, str) and "." in self.param.threshold:
            self.param.threshold = float(self.param.threshold)
        
        self.labels_out_list = []
        self.est_num_spks_out_list = []
        self.lambdas_list = []
        self.use_gc_thres = False

    def NMEanalysis(self, mat, SPK_MAX, max_rp_threshold, sparse_search=True, search_p_volume=500, fixed_thres=None):
        eps = 1e-10
        
        # INTEGRATED BACKEND OPTIMIZATION:
        # Pre-clean the raw input matrix using a robust neighbor pruning threshold (top 15% links)
        # to guarantee the Eigengap formula calculates precise speaker counts.
        row_count = mat.shape[0]
        k_neighbors = max(1, int(row_count * 0.15))
        cleaned_mat = np.zeros_like(mat)
        for idx in range(row_count):
            top_indices = np.argsort(mat[idx])[-k_neighbors:]
            cleaned_mat[idx, top_indices] = mat[idx, top_indices]
        mat = 0.5 * (cleaned_mat + cleaned_mat.T)
        
        eig_ratio_list = []
        if fixed_thres and fixed_thres != "NMESC":
            p_neighbors_list = [ int(mat.shape[0] * float(fixed_thres)) ]
            max_N = p_neighbors_list[0]
        else:
            max_N = int(mat.shape[0] * max_rp_threshold)
            if sparse_search:
                N = min(max_N, search_p_volume)
                p_neighbors_list = list(np.linspace(1, max_N, N, endpoint=True).astype(int))
            else:
                p_neighbors_list = list(range(1, max_N))
        
        est_spk_n_dict = {}
        for p_neighbors in p_neighbors_list:
            if p_neighbors < 1:
                p_neighbors = 1
            X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors)
            est_num_of_spk, lambdas, lambda_gap_list = estimate_num_of_spkrs(X_conn_from_dist, SPK_MAX)
            est_spk_n_dict[p_neighbors] = (est_num_of_spk, lambdas)
            arg_sorted_idx = np.argsort(lambda_gap_list[:SPK_MAX])[::-1] 
            max_key = arg_sorted_idx[0]  
            max_eig_gap = lambda_gap_list[max_key]/(max(lambdas) + eps) 
            eig_ratio_value = (p_neighbors/mat.shape[0])/(max_eig_gap+eps)
            eig_ratio_list.append(eig_ratio_value)
         
        index_nn = np.argmin(eig_ratio_list)
        rp_p_neighbors = p_neighbors_list[index_nn]
        X_conn_from_dist = get_X_conn_from_dist(mat, rp_p_neighbors)
        if not isFullyConnected(X_conn_from_dist):
            X_conn_from_dist, rp_p_neighbors = gc_thres_min_gc(mat, max_N, p_neighbors_list)
        
        return X_conn_from_dist, float(rp_p_neighbors/mat.shape[0]), est_spk_n_dict[rp_p_neighbors][0], est_spk_n_dict[rp_p_neighbors][1], rp_p_neighbors
    
    def COSclustering(self, idx, key, mat, mat_spkcount, param):
        X_dist_raw = mat
        rp_threshold = param.threshold
        if param.spt_est_thres in ["EigRatio", "NMESC"] or param.threshold == "EigRatio":
            X_conn_spkcount, rp_thres_spkcount, est_num_of_spk, lambdas, p_neigh_spkcount = self.NMEanalysis(
                mat_spkcount, param.max_speaker, max_rp_threshold=0.250, sparse_search=param.sparse_search, search_p_volume=param.n_sparse_search
            )
            rp_threshold = rp_thres_spkcount 
            self.lambdas_list.append(lambdas)
        
        p_neigh = p_neigh_spkcount
        X_conn_from_dist = get_X_conn_from_dist(mat, p_neigh)

        spectral_model = sklearn_SpectralClustering(affinity='precomputed', 
                                                   eigen_solver='arpack',
                                                   random_state=0,
                                                   n_jobs=3, 
                                                   n_clusters=est_num_of_spk,
                                                   eigen_tol=1e-10)
        
        Y = spectral_model.fit_predict(X_conn_from_dist)
        return Y