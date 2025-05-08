import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, regularizers

# Add auxiliary function: remove the batch dimension of SparseTensor
def remove_batch_sparse_tensor(sp):
    mask = tf.equal(sp.indices[:, 0], 0)  # Keep only the batch with index 0
    new_indices = sp.indices[mask][:, 1:]  # Remove batch index
    new_values = tf.boolean_mask(sp.values, mask)
    new_dense_shape = sp.dense_shape[1:]
    new_sp = tf.sparse.SparseTensor(new_indices, new_values, new_dense_shape)
    return tf.sparse.reorder(new_sp)


class ContrastiveLearning(tf.keras.layers.Layer):
    def __init__(self, temperature=0.1, projection_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.projection_dim = projection_dim
        
        # Dual-tower projection network
        self.drug_proj = Sequential([
            Dense(256, activation='gelu'),
            LayerNormalization(),
            Dense(projection_dim)
        ])
        self.cell_proj = Sequential([
            Dense(256, activation='gelu'),
            LayerNormalization(),
            Dense(projection_dim)
        ])

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        drug, cell = inputs  # shapes (n_drugs, feat_dim), (n_cells, feat_dim)
        
        # Feature projection + normalization
        drug_proj = tf.math.l2_normalize(self.drug_proj(drug), axis=-1)  # shape (n_drugs, proj_dim)
        cell_proj = tf.math.l2_normalize(self.cell_proj(cell), axis=-1)  # shape (n_cells, proj_dim)

        # Compute similarity matrix (drug-drug and cell-cell)
        drug_sim = tf.matmul(drug_proj, drug_proj, transpose_b=True)  # (n_drugs, n_drugs)
        cell_sim = tf.matmul(cell_proj, cell_proj, transpose_b=True)  # (n_cells, n_cells)

        # -------- Contrastive loss: maximize diagonal similarity, minimize off-diagonal --------
        def contrastive_self_similarity(sim_matrix):
            # shape: (N, N)
            logits = sim_matrix / self.temperature  # softmax logits
            
            labels = tf.eye(tf.shape(sim_matrix)[0])  # shape (N, N)
            loss = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True)
            return tf.reduce_mean(loss)

        drug_loss = contrastive_self_similarity(drug_sim)
        cell_loss = contrastive_self_similarity(cell_sim)

        contrast_loss = (drug_loss + cell_loss) / 2.0
        self.add_loss(contrast_loss)

        # Return sparse similarity matrices (useful for downstream tasks in the main model)
        # --- 1. Flatten & compute top-k selection ---
        flat_drug = tf.reshape(drug_sim, [-1])
        flat_cell = tf.reshape(cell_sim, [-1])
        num_drug = tf.shape(flat_drug)[0]
        num_cell = tf.shape(flat_cell)[0]
        k_drug = tf.maximum(tf.cast(tf.cast(num_drug, tf.float32) * 0.2, tf.int32), 1)
        k_cell = tf.maximum(tf.cast(tf.cast(num_cell, tf.float32) * 0.2, tf.int32), 1)

        # --- 2. Get top-k values, determine threshold ---
        topk_drug = tf.math.top_k(flat_drug, k=k_drug).values
        topk_cell = tf.math.top_k(flat_cell, k=k_cell).values
        thresh_drug = topk_drug[-1]
        thresh_cell = topk_cell[-1]

        # --- 3. Mask & threshold filtering ---
        mask_drug = tf.greater_equal(drug_sim, thresh_drug)
        mask_cell = tf.greater_equal(cell_sim, thresh_cell)
        drug_sim_thresh = tf.where(mask_drug, drug_sim, tf.zeros_like(drug_sim))
        cell_sim_thresh = tf.where(mask_cell, cell_sim, tf.zeros_like(cell_sim))

        # --- 4. Convert to sparse and return ---
        drug_sim_sparse = tf.sparse.from_dense(drug_sim_thresh)
        cell_sim_sparse = tf.sparse.from_dense(cell_sim_thresh)
        return drug_sim_sparse, cell_sim_sparse


class SimilarityEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, start_idx, full_size, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = start_idx
        self.full_size = full_size

    def call(self, sparse_matrix):
        indices = sparse_matrix.indices  # [N, 2]
        values = sparse_matrix.values
        new_indices = indices + tf.constant([[self.start_idx, self.start_idx]], dtype=indices.dtype)
        return tf.SparseTensor(indices=new_indices, values=values, dense_shape=[self.full_size, self.full_size])


class DistMult(Layer):
    def __init__(self, num_relations, embedding_dim, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg

    @tf.autograph.experimental.do_not_convert 
    def call(self, inputs, training=None):
        head_emb, rel_emb, tail_emb = inputs
        score = tf.reduce_sum(head_emb * rel_emb * tail_emb, axis=-1)
        
        return tf.nn.sigmoid(score)


class GraphConvLayer(layers.Layer):
    def __init__(self, output_dim, activation='relu', l2_reg=None, **kwargs):
        self.l2_reg = l2_reg
        super(GraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        regularizer = tf.keras.regularizers.L2(self.l2_reg) if self.l2_reg else None
        # W: weight matrix
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[0][-1], self.output_dim),
            initializer='glorot_uniform',
            regularizer=regularizer,
            trainable=True
        )

    def call(self, inputs):
        node_features, adj_matrix = inputs  # node features and adjacency matrix

        # Graph convolution computation: AXW
        support = tf.matmul(node_features, self.W)  # XW
        output = tf.matmul(adj_matrix, support)  # AXW

        if self.activation == 'relu':
            output = tf.nn.relu(output)
        elif self.activation == 'gelu':
            output = tf.nn.gelu(output)

        return output
    
class DrugFeatureExtractor(layers.Layer):
    def __init__(self, output_dim, dropout_rate, units_list=[128, 64], l2_reg=None, **kwargs):
        super().__init__(**kwargs)
        self.units_list = units_list
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # Add graph convolution layers
        self.graph_conv_layers = [
            GraphConvLayer(units, activation='gelu', l2_reg=self.l2_reg)
            for units in self.units_list
        ]
        # Final graph convolution layer
        self.final_graph_conv = GraphConvLayer(self.output_dim, activation=None, l2_reg=self.l2_reg)

        # Dropout and LayerNormalization
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()

        self.global_pool = layers.GlobalAveragePooling1D()
        
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        node_features, adj_matrix = inputs  # Input contains node features and adjacency matrix
        x = node_features
        
        # Graph convolution operation
        for layer in self.graph_conv_layers:
            x = layer([x, adj_matrix])  # Perform graph convolution using the adjacency matrix
            x = self.dropout(x, training=training)
        
        # Final graph convolution layer
        x = self.final_graph_conv([x, adj_matrix])
        x = self.layer_norm(x)

        x = self.global_pool(x)

        return x


class CellFeatureExtractor(Layer):
    def __init__(self, output_dim, dropout_rate=0.2, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        
        # ========== Gene expression processing ==========
        self.gexpr_main = Sequential([
            Dense(256, activation='gelu', kernel_regularizer=l2(l2_reg)),
            Dropout(dropout_rate),
            Dense(output_dim, kernel_regularizer=l2(l2_reg)),  # Final output 128-dimensional
            LayerNormalization()
        ])
        # ========== Copy number processing ==========
        self.copy_proj = Sequential([
            Dense(256, activation='gelu', kernel_regularizer=l2(l2_reg)),
            Dropout(dropout_rate),
            Dense(output_dim, kernel_regularizer=l2(l2_reg)),  # Final output 128-dimensional
            LayerNormalization()
        ])
        # ========== Mutation processing ==========
        self.mut_conv = Sequential([
            Reshape((689, 1), input_shape=(689,)),
            Conv1D(output_dim*3, 50, strides=5, activation='gelu'),
            MaxPooling1D(pool_size=5, strides=5),
            Conv1D(512, 5, strides=2, activation='gelu'),
            GlobalAveragePooling1D(),
            Dense(output_dim),
            LayerNormalization()
        ])
        
        # ========== Feature fusion ==========
        self.fusion = Dense(output_dim, activation='gelu')
        self.final_ln = LayerNormalization()
        
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        gexpr, copy, mut = inputs
        
        # ========== Gene expression processing ==========
        gexpr_x = self.gexpr_main(gexpr)

        # ========== Copy number processing ==========
        copy_x = self.copy_proj(copy)

        # ========== Mutation processing ==========
        mut_x = self.mut_conv(mut)

        # ========== Feature fusion ==========
        combined = self.fusion(gexpr_x + copy_x + mut_x)
        return self.final_ln(combined)
    
class DGCNLayer(Layer):
    def __init__(self, 
                 embedding_dim=128,
                 num_relations=4,  # Sensitivity/Resistance/Drug similarity/Cell similarity
                 dropout_rate=0.1,
                 l2_reg=1e-4,
                 use_contrastive_gating=True,  # New: Use contrastive learning gating
                 res_mats_dim = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_contrastive_gating = use_contrastive_gating
        self.res_mats_dim = res_mats_dim
        
    def build(self, input_shape):
        # 1. Relation-specific parameters
        self.relation_kernels = [
            Dense(self.embedding_dim,
                 kernel_regularizer=l2(self.l2_reg),
                 name=f'relation_kernel_{i}')
            for i in range(self.num_relations)
        ]
        # 2. Shared layer for fusion of relation embeddings and node features
        self.rel_fuse_dense = Dense(self.embedding_dim, activation='relu')

        # 3. Dynamic relation weights (including contrastive learning adjustment)
        self.relation_weights = self.add_weight(
            name='relation_weights',
            shape=(self.num_relations,),
            initializer=tf.keras.initializers.Constant(1.0),
            constraint=tf.keras.constraints.NonNeg(),  # Non-negative weights
            trainable=True
        )
        
        # 4. Gating mechanism enhancement
        self.gate_kernel = Dense(self.embedding_dim, activation='sigmoid')
        
        # 4. Contrastive learning gating (new addition)
        if self.use_contrastive_gating:
            self.contrastive_gate = Dense(self.embedding_dim, activation='sigmoid')
        
        self.dropout = Dropout(self.dropout_rate)
        self.ln = LayerNormalization()

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        # Unpack inputs
        node_embeddings, adjacency_mats, rel_embeddings  = inputs
        adjacency_mats = [tf.sparse.reshape(adjacency_mats[i], shape=self.res_mats_dim) if i < 2 else adjacency_mats[i] for i in range(4)]
        contrast_features = None

        relation_outputs = []
        # --- Multi-relation message passing ---
        for rel_idx in range(self.num_relations):
            messages = tf.sparse.sparse_dense_matmul(
                adjacency_mats[rel_idx], 
                node_embeddings
            )
            # Fuse relation embeddings and node information (method 1, concatenation)
            rel_vec = rel_embeddings[rel_idx]  # shape: [embedding_dim]
            rel_vec_broadcast = tf.broadcast_to(rel_vec, tf.shape(messages))  # shape: [1, embedding_dim]
            rel_concat = tf.concat([messages, rel_vec_broadcast], axis=-1)
            rel_biased = self.rel_fuse_dense(rel_concat)
            transformed = self.relation_kernels[rel_idx](rel_biased)
            relation_outputs.append(transformed)
        
        # --- Dynamic weight adjustment ---
        weights = tf.nn.softmax(self.relation_weights)  # shape: [num_relations]
        relation_outputs_tensor = tf.stack(relation_outputs, axis=0)  # shape: [num_relations, nodes, features]
        weights_expanded = tf.reshape(weights, [self.num_relations, 1, 1])  # shape: [num_relations, 1, 1]
        combined = tf.reduce_sum(weights_expanded * relation_outputs_tensor, axis=0)  # shape: [nodes, features]
        
        # --- Gating enhancement module ---
        base_gate = self.gate_kernel(node_embeddings)
        
        # Contrastive learning gating (new addition)
        if self.use_contrastive_gating and contrast_features is not None:
            contrast_gate = self.contrastive_gate(contrast_features)
            final_gate = 0.7 * base_gate + 0.3 * contrast_gate  # Adjustable ratio
        else:
            final_gate = base_gate
        
        # --- Residual connection and output ---
        output = self.ln(
            node_embeddings + final_gate * combined
        )
        output = self.dropout(output, training=training)

        return output

class CustomTrainer(tf.keras.Model):
    def __init__(self, model, contrastive_module, loss_fn, optimizer, main_loss_weight=1.0, contrast_loss_weight=0.1):
        super().__init__()
        self.model = model
        self.contrastive_module = contrastive_module
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.main_loss_weight = main_loss_weight
        self.contrast_loss_weight = contrast_loss_weight

        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")

    @property
    def metrics(self):
        return [self.train_loss_tracker]

    def train_step(self, data):
        inputs, labels = data

        # === unpack inputs ===
        (
            head_input, rel_input, tail_input,
            gexpr_input, copy_input, mut_input,
            atom_feat_input, atom_adj_input,
            response_adj1, response_adj2
        ) = inputs

        with tf.GradientTape() as tape:
            # === 1. Main task prediction ===
            rel_pred = self.model(inputs, training=True)
            main_loss = self.loss_fn(labels, rel_pred)

            # === 2. Contrastive loss (drug-cell inputs) ===
            # You need to pass in processed_drug, cell_features (assumed to be pre-calculated)
            drug_features = processed_drug   # shape = (num_drugs, feature_dim)
            cell_features = tf.concat([gexpr_input, copy_input, mut_input], axis=-1)

            drug_sim_sparse, cell_sim_sparse = self.contrastive_module([drug_features, cell_features], training=True)
            
            # Contrastive module already adds loss internally, just use its .losses
            contrast_loss = tf.add_n(self.contrastive_module.losses)

            # === 3. Multi-task loss fusion ===
            total_loss = self.main_loss_weight * main_loss + self.contrast_loss_weight * contrast_loss

        # === 4. Backpropagation ===
        gradients = tape.gradient(total_loss, self.model.trainable_variables + self.contrastive_module.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables + self.contrastive_module.trainable_variables))

        self.train_loss_tracker.update_state(total_loss)

        return {"loss": self.train_loss_tracker.result()}
