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
        """加载原始数据"""
        pub_smiles = pd.read_csv(self.config.SMILES_PATH)
        mutation = pd.read_csv(self.config.MUTATION_PATH)
        gexpr = pd.read_csv(self.config.GEXPR_PATH)
        copy_number = pd.read_csv(self.config.COPY_NUMBER_PATH)
        response = pd.read_csv(self.config.RESPONSE_PATH)
        return pub_smiles, mutation, gexpr, copy_number, response
        
    def calculate_graph_feat(self, feat_mat, adj_list):
        """计算分子图特征"""
        adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
        for i, nodes in enumerate(adj_list):
            for each in nodes:
                adj_mat[i, int(each)] = 1
        assert np.allclose(adj_mat, adj_mat.T)
        x, y = np.where(adj_mat == 1)
        adj_index = np.array(np.vstack((x, y)))
        return [feat_mat, adj_index]

    def feature_extract(self, drug_feature):
        """提取药物特征"""
        drug_data = []
        for i in range(len(drug_feature)):
            feat_mat, adj_list = drug_feature.iloc[i]
            drug_data.append(self.calculate_graph_feat(feat_mat, adj_list))
        return drug_data

    def cmask(self, num, ratio, seed):
        """创建训练测试掩码"""
        mask = np.ones(num, dtype=bool)
        mask[0:int(ratio * num)] = False
        np.random.seed(seed)
        np.random.shuffle(mask)
        return mask

    def pad_features(self, features_list):
        """对分子特征进行填充，使所有分子具有相同数量的原子
        
        Args:
            features_list: 包含所有分子特征矩阵的列表
            
        Returns:
            padded_features: 填充后的特征矩阵
            padded_adj: 填充后的邻接矩阵
        """
        # 找出最大原子数
        max_atoms = max(feat[0].shape[0] for feat in features_list)
        feat_dim = features_list[0][0].shape[1]
        
        # 初始化填充后的数组
        num_mols = len(features_list)
        padded_features = np.zeros((num_mols, max_atoms, feat_dim), dtype=np.float32)
        padded_adj = np.zeros((num_mols, max_atoms, max_atoms), dtype=np.float32)
        
        # 填充每个分子的特征
        for i, (feat, adj) in enumerate(features_list):
            n_atoms = feat.shape[0]
            padded_features[i, :n_atoms, :] = feat
            # 处理邻接矩阵
            adj_dense = np.zeros((max_atoms, max_atoms), dtype=np.float32)
            if adj.size > 0:  # 确保邻接矩阵不为空
                adj_dense[:n_atoms, :n_atoms][adj[0], adj[1]] = 1
            padded_adj[i] = adj_dense
            
        return padded_features, padded_adj


    def process_data(self, pub_smiles, mutation_feature, gexpr_feature, copy_number_feature, data_new, test_size=0.1):
        """处理数据，按9:1划分训练+验证集和测试集"""

        # 获取有效的pubmedid（有SMILES的药物ID）
        pubmedid = list(set([int(item[4]) for item in data_new if int(item[4]) in pub_smiles['pubchems'].values]))
        pubmedid.sort()
        
        # 用有效的pubmedid过滤data_new
        filtered_data = [item for item in data_new if int(item[4]) in pubmedid]
        
        # 从过滤后的数据中获取cellineid
        cellineid = list(set([item[0] for item in filtered_data]))
        cellineid.sort()
        
        # 构建节点映射
        cellmap = list(zip(cellineid, list(range(len(cellineid)))))
        pubmedmap = list(zip(pubmedid, list(range(len(cellineid), len(cellineid) + len(pubmedid)))))

        # 获取数值索引（使用过滤后的数据）
        cellline_num = np.squeeze([[j[1] for j in cellmap if i[0] == j[0]] for i in filtered_data])
        pubmed_num = np.squeeze([[j[1] for j in pubmedmap if int(i[4]) == j[0]] for i in filtered_data])
        rel = np.squeeze([i[3] for i in filtered_data])
        
        # 构建三元组 (头实体, 关系, 尾实体)
        allpairs = np.vstack((cellline_num, rel, pubmed_num)).T
        allpairs = allpairs[allpairs[:, 1].argsort()]  # 按关系排序

        node_num = len(cellmap) + len(pubmedmap)

        # 处理药物特征
        drug_feature = {}
        featurizer = dc.feat.ConvMolFeaturizer()
        for pubchem_id, isosmiles in zip(pub_smiles['pubchems'], pub_smiles['isosmiles']):
            mol = Chem.MolFromSmiles(isosmiles)
            X = featurizer.featurize(mol)
            drug_feature[str(int(pubchem_id))] = [X[0].get_atom_features(), X[0].get_adjacency_list()]

        # 获取药物ID并确保类型匹配
        pubid = [str(item[0]) for item in pubmedmap]
        drug_feature = pd.DataFrame(drug_feature).T
        drug_feature = drug_feature.loc[pubid]
        atom_shape = drug_feature.iloc[0][0].shape[-1]
        drug_data = self.feature_extract(drug_feature)

        # 处理drug_features
        if isinstance(drug_data, list):
            atom_features, adj_matrices = self.pad_features(drug_data)
            atom_features = tf.convert_to_tensor(atom_features, dtype=tf.float32)
            adj_matrices = tf.convert_to_tensor(adj_matrices, dtype=tf.float32)
            drug_data = {
                'atom_features': atom_features,
                'adjacency_matrix': adj_matrices
            }

        # 处理细胞系特征
        cellid = [item[0] for item in cellmap]
        gexpr_feature = gexpr_feature.set_index('cell_line').loc[cellid]
        mutation_feature = mutation_feature.set_index('cell_line').loc[cellid]
        copy_number_feature = copy_number_feature.set_index('cell_line').loc[cellid]

        mutation = tf.convert_to_tensor(np.array(mutation_feature, dtype='float32'))
        gexpr = tf.convert_to_tensor(np.array(gexpr_feature, dtype='float32'))
        copy_number = tf.convert_to_tensor(np.array(copy_number_feature, dtype='float32'))

        # 按9:1划分训练+验证集 和 测试集
        train_val_indices, test_indices = train_test_split(
            np.arange(allpairs.shape[0]),
            test_size=test_size,
            random_state=self.config.SEED
        )
        train_val_pairs = allpairs[train_val_indices]
        test_pairs = allpairs[test_indices]

        # 只返回 9:1 划分和特征
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
        """加载并处理所有数据"""
        # 加载原始数据
        pub_smiles, mutation, gexpr, copy_number, response = self.load_raw_data()
        
        # 处理数据
        processed_data = self.process_data(pub_smiles, mutation, gexpr, copy_number, response.values)
        
        
        # 准备邻接矩阵
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
        """获取邻接矩阵"""

        if X_train.shape[0] == 0:
            return []

        # 假设 X_train 每一行格式为 [source, relation, target]
        num_entities = node_num  # 获取最大实体数，假设节点编号从 0 开始

        # 提取所有唯一关系类型
        unique_rels = np.unique(X_train[:, 1])

        # 为每种关系累计所有边
        rel_edges = {r: ([], []) for r in unique_rels}
        for edge in X_train:
            src, r, dst = edge
            src = int(src)
            dst = int(dst)
            r = int(r)
            rel_edges[r][0].append(src)
            rel_edges[r][1].append(dst)

        # 构造每个关系对应的邻接矩阵
        adj_mats = []
        for r in unique_rels:
            rows = np.array(rel_edges[r][0])
            cols = np.array(rel_edges[r][1])
            
            # 处理重复索引：将索引和对应的值收集到字典中
            edge_dict = {}
            for i in range(len(rows)):
                key = (rows[i], cols[i])
                if key not in edge_dict:
                    edge_dict[key] = 0
                edge_dict[key] += 1  # 累加重复边的权重
            
            # 从字典重建唯一的行、列和值数组
            unique_rows = []
            unique_cols = []
            unique_data = []
            for (r, c), weight in edge_dict.items():
                unique_rows.append(r)
                unique_cols.append(c)
                unique_data.append(float(weight))  # 使用权重作为边的值
            
            # 构造稀疏张量
            indices = np.column_stack((unique_rows, unique_cols))
            values = np.array(unique_data, dtype=np.float32)
            
            sparse_tensor = tf.sparse.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=(num_entities, num_entities)
            )
            
            # 使用tf.sparse.reorder确保索引已排序
            sparse_tensor = tf.sparse.reorder(sparse_tensor)
            
            adj_mats.append(sparse_tensor)

        return adj_mats
