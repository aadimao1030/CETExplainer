import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
import deepchem as dc
from sklearn.model_selection import KFold, train_test_split

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_raw_data(self):
        """Load raw data"""
        pub_smiles = pd.read_csv(self.config.SMILES_PATH)
        mutation = pd.read_csv(self.config.MUTATION_PATH)
        gexpr = pd.read_csv(self.config.GEXPR_PATH)
        copy_number = pd.read_csv(self.config.COPY_NUMBER_PATH)
        response = pd.read_csv(self.config.RESPONSE_PATH)
        return pub_smiles, mutation, gexpr, copy_number, response
        
    def calculate_graph_feat(self, feat_mat, adj_list):
        """Calculate molecular graph features"""
        adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
        for i, nodes in enumerate(adj_list):
            for each in nodes:
                adj_mat[i, int(each)] = 1
        assert np.allclose(adj_mat, adj_mat.T)
        x, y = np.where(adj_mat == 1)
        adj_index = np.array(np.vstack((x, y)))
        return [feat_mat, adj_index]

    def feature_extract(self, drug_feature):
        """Extract drug features"""
        drug_data = []
        for i in range(len(drug_feature)):
            feat_mat, adj_list = drug_feature.iloc[i]
            drug_data.append(self.calculate_graph_feat(feat_mat, adj_list))
        return drug_data

    def cmask(self, num, ratio, seed):
        """Create training and testing masks"""
        mask = np.ones(num, dtype=bool)
        mask[0:int(ratio * num)] = False
        np.random.seed(seed)
        np.random.shuffle(mask)
        return mask

    def pad_features(self, features_list):
        """Pad molecular features to ensure all molecules have the same number of atoms
        
        Args:
            features_list: A list containing feature matrices for all molecules
            
        Returns:
            padded_features: Padded feature matrices
            padded_adj: Padded adjacency matrices
        """
        # Find the maximum number of atoms
        max_atoms = max(feat[0].shape[0] for feat in features_list)
        feat_dim = features_list[0][0].shape[1]
        
        # Initialize padded arrays
        num_mols = len(features_list)
        padded_features = np.zeros((num_mols, max_atoms, feat_dim), dtype=np.float32)
        padded_adj = np.zeros((num_mols, max_atoms, max_atoms), dtype=np.float32)
        
        # Pad features for each molecule
        for i, (feat, adj) in enumerate(features_list):
            n_atoms = feat.shape[0]
            padded_features[i, :n_atoms, :] = feat
            # Handle adjacency matrix
            adj_dense = np.zeros((max_atoms, max_atoms), dtype=np.float32)
            if adj.size > 0:  # Ensure adjacency matrix is not empty
                adj_dense[:n_atoms, :n_atoms][adj[0], adj[1]] = 1
            padded_adj[i] = adj_dense
            
        return padded_features, padded_adj


    def process_data(self, pub_smiles, mutation_feature, gexpr_feature, copy_number_feature, data_new, test_size=0.1):
        """Process data, split into 90% training+validation and 10% test"""
        
        # Get valid pubmedid (drug IDs with SMILES)
        pubmedid = list(set([int(item[4]) for item in data_new if int(item[4]) in pub_smiles['pubchems'].values]))
        pubmedid.sort()
        
        # Filter data_new by valid pubmedid
        filtered_data = [item for item in data_new if int(item[4]) in pubmedid]
        
        # Get cellineid from filtered data
        cellineid = list(set([item[0] for item in filtered_data]))
        cellineid.sort()
        
        # Build node mapping
        cellmap = list(zip(cellineid, list(range(len(cellineid)))))
        pubmedmap = list(zip(pubmedid, list(range(len(cellineid), len(cellineid) + len(pubmedid)))))
        
        # Get numeric indices using filtered data
        cellline_num = np.squeeze([[j[1] for j in cellmap if i[0] == j[0]] for i in filtered_data])
        pubmed_num = np.squeeze([[j[1] for j in pubmedmap if int(i[4]) == j[0]] for i in filtered_data])
        rel = np.squeeze([i[3] for i in filtered_data])
        
        # Build triplets (head entity, relation, tail entity)
        allpairs = np.vstack((cellline_num, rel, pubmed_num)).T
        allpairs = allpairs[allpairs[:, 1].argsort()]  # Sort by relation
        
        node_num = len(cellmap) + len(pubmedmap)

        # Process drug features
        drug_feature = {}
        featurizer = dc.feat.ConvMolFeaturizer()
        for pubchem_id, isosmiles in zip(pub_smiles['pubchems'], pub_smiles['isosmiles']):
            mol = Chem.MolFromSmiles(isosmiles)
            X = featurizer.featurize(mol)
            drug_feature[str(int(pubchem_id))] = [X[0].get_atom_features(), X[0].get_adjacency_list()]

        # Get drug IDs and ensure type match
        pubid = [str(item[0]) for item in pubmedmap]
        drug_feature = pd.DataFrame(drug_feature).T
        drug_feature = drug_feature.loc[pubid]
        atom_shape = drug_feature.iloc[0][0].shape[-1]
        drug_data = self.feature_extract(drug_feature)

        # Process drug features
        if isinstance(drug_data, list):
            atom_features, adj_matrices = self.pad_features(drug_data)
            atom_features = tf.convert_to_tensor(atom_features, dtype=tf.float32)
            adj_matrices = tf.convert_to_tensor(adj_matrices, dtype=tf.float32)
            drug_data = {
                'atom_features': atom_features,
                'adjacency_matrix': adj_matrices
            }

        # Process cell line features
        cellid = [item[0] for item in cellmap]
        gexpr_feature = gexpr_feature.set_index('cell_line').loc[cellid]
        mutation_feature = mutation_feature.set_index('cell_line').loc[cellid]
        copy_number_feature = copy_number_feature.set_index('cell_line').loc[cellid]

        mutation = tf.convert_to_tensor(np.array(mutation_feature, dtype='float32'))
        gexpr = tf.convert_to_tensor(np.array(gexpr_feature, dtype='float32'))
        copy_number = tf.convert_to_tensor(np.array(copy_number_feature, dtype='float32'))

        # Split data into 90% training+validation and 10% testing
        train_val_indices, test_indices = train_test_split(
            np.arange(allpairs.shape[0]),
            test_size=test_size,
            random_state=self.config.SEED
        )
        train_val_pairs = allpairs[train_val_indices]
        test_pairs = allpairs[test_indices]

        # Return 90:10 split and features
        data = {
            'train_edge': train_val_pairs,
            'test_edge': test_pairs,
            'drug_data': drug_data,
            'mutation': mutation,
            'gexpr': gexpr,
            'copy_number': copy_number,
            'atom_shape': atom_shape,
            'node_num': node_num
        }

        return data

    def load_data(self):
        """Load and process all data"""
        # Load raw data
        pub_smiles, mutation, gexpr, copy_number, response = self.load_raw_data()
        
        # Process data
        processed_data = self.process_data(pub_smiles, mutation, gexpr, copy_number, response.values)
        
        # Prepare adjacency matrices
        ADJ_MATS = self._get_adj_mats(processed_data['train_edge'], processed_data['node_num'])
        
        return {
            'X_train': processed_data['train_edge'],
            'X_test': processed_data['test_edge'],
            'gexpr_data': processed_data['gexpr'],
            'copy_number_data': processed_data['copy_number'],
            'mutation_data': processed_data['mutation'],
            'ADJ_MATS': ADJ_MATS,
            'drug_data': processed_data['drug_data']
        }

    def _get_adj_mats(self, X_train, node_num):
        """Get adjacency matrices"""

        if X_train.shape[0] == 0:
            return []

        # Assuming X_train format: [source, relation, target]
        num_entities = node_num  # Get the maximum number of entities, assuming node IDs start from 0

        # Extract all unique relation types
        unique_rels = np.unique(X_train[:, 1])

        # Accumulate all edges for each relation
        rel_edges = {r: ([], []) for r in unique_rels}
        for edge in X_train:
            src, r, dst = edge
            src = int(src)
            dst = int(dst)
            r = int(r)
            rel_edges[r][0].append(src)
            rel_edges[r][1].append(dst)

        # Construct adjacency matrices for each relation
        adj_mats = []
        for r in unique_rels:
            rows = np.array(rel_edges[r][0])
            cols = np.array(rel_edges[r][1])
            
            # Handle duplicate indices: accumulate weights for edges
            edge_dict = {}
            for i in range(len(rows)):
                key = (rows[i], cols[i])
                if key not in edge_dict:
                    edge_dict[key] = 0
                edge_dict[key] += 1  # Accumulate weight for duplicate edges
            
            # Rebuild unique rows, cols, and data arrays from the dictionary
            unique_rows = []
            unique_cols = []
            unique_data = []
            for (r, c), weight in edge_dict.items():
                unique_rows.append(r)
                unique_cols.append(c)
                unique_data.append(float(weight))  # Use weight as edge value
            
            # Construct sparse tensor
            indices = np.column_stack((unique_rows, unique_cols))
            values = np.array(unique_data, dtype=np.float32)
            
            sparse_tensor = tf.sparse.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=(num_entities, num_entities)
            )
            
            # Reorder sparse tensor to ensure indices are sorted
            sparse_tensor = tf.sparse.reorder(sparse_tensor)
            
            adj_mats.append(sparse_tensor)

        return adj_mats
