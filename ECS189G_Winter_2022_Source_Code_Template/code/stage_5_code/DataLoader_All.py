import pandas as pd
from code.base_class.dataset import dataset
import scipy.sparse as sp
import numpy as np
import torch
from collections import Counter

data_path = "/Users/jacquelinemitchell/Documents/ECS189G/sample_code/" \
            "ECS189G-21W/ECS189G_Winter_2022_Source_Code_Template/data/stage_5_data/cora"


class Dataset_Loader(dataset):

    data_to_node_features = {'cora' : 1433, 'citeseer' : 3703, 'pubmed' : 500}

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dName = dName
        assert self.dName is not None

    def load_raw(self):
        path = self.dataset_source_folder_path + f"/{self.dName}"
        num_features = self.data_to_node_features[self.dName]

        column_names_node = ["Node"] + [f"word_{i}" for i in range(num_features)] + ["category"]
        node_data = pd.read_csv(data_path + "/node", sep="\t", engine="python", header=None, names=column_names_node)

        column_names_edges = ["target", "source"]
        edge_data = pd.read_csv(data_path + "/link", sep="\t", engine="python", header=None, names=column_names_edges)

        class_dict = {c : i for i, c in enumerate(node_data.category.unique())}
        node_data['category'] = [class_dict[i] for i in node_data['category']]

        print(node_data.shape)

        return node_data.to_numpy(), edge_data.to_numpy(), class_dict

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_training_testing_idx(self, possible_idx, num_classes, categorial_labels):

        train_indices = []
        for c in range(num_classes):
            chosen_indices = list(
                np.random.choice(possible_idx[np.where(categorial_labels[possible_idx] == c)[0]], 20,
                                 replace=False))
            train_indices += chosen_indices

        test_indices = []
        valid_idx = np.array(list(set(list(possible_idx)) - set(train_indices)))
        for c in range(num_classes):

            if len(valid_idx[np.where(categorial_labels[valid_idx] == c)[0]]) <= 200:
                test_indices += list(valid_idx[np.where(categorial_labels[valid_idx] == c)[0]])
            else:
                chosen_indices = np.random.choice(valid_idx[np.where(categorial_labels[valid_idx] == c)[0]],
                                                  200, replace=False)
                test_indices += list(chosen_indices)

        assert len(list(set(train_indices) - set(test_indices))) == len(train_indices)

        return train_indices, test_indices

    def load(self, node_data, edge_data):

        # Node features
        features = sp.csr_matrix(node_data[:, 1:-1], dtype=np.float32)

        # Not going to OHE, because when using PyTorch CE Loss, we don't need to do this step
        categorial_labels = np.array(node_data[:, -1], dtype=np.int32)

        idx = np.array(node_data[:, 0], dtype=np.int32)
        idx_map = {j : i for i, j in enumerate(idx)}

        idx_map_reversed = {i : j for i, j in enumerate(idx)}

        edges_unordered = np.array(edge_data, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(categorial_labels.shape[0], categorial_labels.shape[0]), dtype=np.float32)

        # We are just getting the augmented A matrix (with 1s along main diagonal too and 1 where each edge is conn
        # ected, but it is represented sparsly in memory?
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(categorial_labels)
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        possible_idx = np.array(range(features.size()[0]))
        if self.dName == 'cora':
            num_classes = 7
            train_indices, test_indices = self.get_training_testing_idx(possible_idx, num_classes, categorial_labels)
        elif self.dName == 'citeseer':
            num_classes = 6
            train_indices, test_indices = self.get_training_testing_idx(possible_idx, num_classes, categorial_labels)
        elif self.dName == 'pubmed':
            num_classes = 3
            train_indices, test_indices = self.get_training_testing_idx(possible_idx, num_classes, categorial_labels)
        else:
            print("Not a valid dataset.")
            assert False == True

        idx_train = torch.LongTensor(np.array(train_indices))
        idx_test = torch.LongTensor(np.array(test_indices))

        train_test = {'idx_train': idx_train, 'idx_test': idx_test}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': categorial_labels,
                 'utility': {'A': adj, 'reverse_idx': idx_map_reversed}}
        return {'graph': graph, 'train_test_val': train_test}












