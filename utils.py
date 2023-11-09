#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
from torch.utils.data import random_split
import torch_geometric
from torch_geometric.utils import to_networkx

import numpy as np
import pandas as pd
import dgl
from torch_sparse import SparseTensor
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
from torch_geometric.utils import remove_self_loops, add_self_loops

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




def init_edge(data):
    assert data.edge_index.shape[0] == 2
    edges = dgl.graph((data.edge_index[0], data.edge_index[1]))
    edges = edges.add_self_loop().remove_self_loop()
    edges = [a.long() for a in edges.edges()]
    edge_index = torch_geometric.data.Data(edge_index=torch.stack(edges)).edge_index
    adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :], sparse_sizes=(data.num_nodes, data.num_nodes))
    return adj


def generate_split(num_samples: int, train_ratio: float, val_ratio: float, labels, ignore_negative= True):

    if ignore_negative:
        labeled_nodes = torch.where(labels != -1)[0]
        labeled_nodes_num = labeled_nodes.shape[0]
    else:
        labeled_nodes = labels
        labeled_nodes_num = num_samples


    train_len = int(labeled_nodes_num * train_ratio)
    val_len = int(labeled_nodes_num * val_ratio)
    test_len = labeled_nodes_num - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, labeled_nodes_num), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    idx_train = labeled_nodes[idx_train]
    idx_val = labeled_nodes[idx_val]
    idx_test = labeled_nodes[idx_test]

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


def compute_semk_hop_new(features, adj, name):
    sims = cosine_similarity(features.cpu())
    adj = adj.cpu()

    adj_tmp = adj#cpu
    sem_khop = adj_tmp
    # sem_khop_tmp = sparse_mx_to_torch_sparse_tensor(sem_khop)
    # padding = torch.zeros_like(sem_khop_tmp)
    padding = torch.zeros_like(sem_khop)
    # adj_sem = torch.where(sem_khop_tmp.to_dense() > 1e-8, torch.FloatTensor(sims), padding.to_dense())
    adj_sem = torch.where(sem_khop.to_dense() > 1e-8, torch.FloatTensor(sims),
                          padding.to_dense())
    adj_sem = adj_sem - torch.diag(torch.diag(adj_sem))
    adj_sem = normalize_adj(adj_sem)
    torch.save(adj_sem, f'saved/{name}_1_hop_new.pt')

    for sem_k in range(2, 7):
        adj_tmp = torch.sparse.mm(adj_tmp, adj)
        sem_khop = sem_khop + adj_tmp
        # sem_khop_tmp = sparse_mx_to_torch_sparse_tensor(sem_khop)
        adj_sem = torch.where(sem_khop.to_dense() > 1e-8,
                              torch.FloatTensor(sims), padding.to_dense())
        adj_sem = adj_sem - torch.diag(torch.diag(adj_sem))
        adj_sem = normalize_adj(adj_sem)
        torch.save(adj_sem, f'saved/{name}_{sem_k}_hop_new.pt')



def process_row(row_idx, values, indices, k):
    topk_idx = values.argsort(descending=True)[:k]
    topk_val = values[topk_idx]
    topk_idx = torch.LongTensor(indices[0, topk_idx])
    row_idx = torch.LongTensor([row_idx] * len(topk_idx))
    return row_idx, topk_idx, topk_val

def get_topk_idx_multithread(k, sparse_matrix):

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        topk_row_idx = []
        topk_indices = []
        topk_values = []

        for i in range(sparse_matrix.size(0)):
            rdata = sparse_matrix[i]
            values, indices = rdata.coalesce().values(), rdata.coalesce().indices()
            futures.append(executor.submit(process_row, i, values, indices, k))

        # 收集结果
        for future in futures:
            row_idx, topk_idx, topk_val = future.result()
            topk_row_idx.append(row_idx)
            topk_indices.append(topk_idx)
            topk_values.append(topk_val)
    executor.shutdown()
    row = torch.cat(topk_row_idx, dim=0)
    col = torch.cat(topk_indices, dim=0)
    return row, col

def compute_semk_hop_new_kn(data, name, semknn):#adj not normalize
    sims = cosine_similarity(data.x.cpu())
    num_nodes = data.x.size(0)
    edge_index = data.edge_index.cpu()
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index, None)

    row, col = edge_index
    row = row - row.min()  # for sampling
    adj = SparseTensor(row=row, col=col, sparse_sizes=(
        num_nodes, num_nodes)).to_torch_sparse_coo_tensor()

    adj_tmp = adj#cpu
    #sem_khop = adj_tmp
    # sem_khop_tmp = sparse_mx_to_torch_sparse_tensor(sem_khop)
    # padding = torch.zeros_like(sem_khop_tmp)
    rows, columns = get_topk_idx_multithread(semknn, adj_tmp)
    edge_index1 = torch.stack([rows, columns], dim=0)
    edge_index1, _ = add_self_loops(edge_index1, num_nodes=num_nodes)
    rows, columns = edge_index1
    sims_values = sims[rows, columns]
    #indices = torch.stack([rows, columns], dim=0)
    adj_sem_tmp_without_normalize = sp.coo_matrix((sims_values, (rows, columns)), shape=(num_nodes, num_nodes))
    adj_sem_tmp = normalize_adj(adj_sem_tmp_without_normalize)
    adj_sem_tmp_without_normalize = sparse_mx_to_torch_sparse_tensor(adj_sem_tmp_without_normalize)
    adj_sem = sparse_mx_to_torch_sparse_tensor(adj_sem_tmp)
    #adj_sem = torch.sparse_coo_tensor(indices=indices, values=sims_values, size=[num_nodes, num_nodes])
    torch.save(adj_sem, f'saved/{name}_1_hop_new_{semknn}k.pt')
    #torch.save(adj_sem_tmp_without_normalize, f'saved/{name}_1_hop_new_{semknn}k.pt')
    for sem_k in range(2, 7):
        adj_tmp = torch.sparse.mm(adj_tmp, adj)
        edge_index = adj_tmp.coalesce().indices()
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index, _ = remove_self_loops(edge_index, None)
        row, col = edge_index
        row = row - row.min()
        adj_tmp_new = SparseTensor(row=row, col=col, sparse_sizes=(
            num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        rows, columns = get_topk_idx_multithread(semknn, adj_tmp_new)
        edge_index1 = torch.stack([rows, columns], dim=0)
        edge_index1, _ = add_self_loops(edge_index1, num_nodes=num_nodes)
        rows, columns = edge_index1
        adj_tmp_k = SparseTensor(row=rows, col=columns, sparse_sizes=(
                num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        adj_sem_new = adj_sem_tmp_without_normalize + adj_tmp_k
        rows, columns = adj_sem_new.coalesce().indices()
        sims_values = sims[rows, columns]
        #indices = torch.stack([rows, columns], dim=0)
        adj_sem_tmp_without_normalize = sp.coo_matrix((sims_values, (rows, columns)), shape=(num_nodes, num_nodes))
        adj_sem_tmp = normalize_adj(adj_sem_tmp_without_normalize)
        adj_sem_tmp_without_normalize = sparse_mx_to_torch_sparse_tensor(adj_sem_tmp_without_normalize)
        adj_sem = sparse_mx_to_torch_sparse_tensor(adj_sem_tmp)
        #adj_sem = torch.sparse_coo_tensor(indices=indices, values=sims_values, size=[num_nodes, num_nodes])
        torch.save(adj_sem, f'saved/{name}_{sem_k}_hop_new_{semknn}k.pt')
        #torch.save(adj_sem_tmp_without_normalize, f'saved/{name}_{sem_k}_hop_new_{semknn}k.pt')
