import tensorflow as tf
import numpy as np
from utils import (generate_negative_samples, build_adj_from_edges, 
                   sparse_tensor_to_triples, compute_metrics,compute_loss, compute_contrastive_loss)
from model_build import build_model
from layers import DistMult
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def init_dummy_run(num_nodes, num_cells, config, data):
    """
    Initialize model components and perform a dummy forward pass to build weights.
    Returns instantiated components and optimizer, trainable variable list.
    """
    drug_ex, cell_ex, contrastive, drug_sim, cell_sim, dgcn1, dgcn2, rel_emb = \
        build_model(num_nodes, num_cells, config)
    distmult = DistMult(config.NUM_RELATIONS, config.EMBEDDING_DIM)

    # Dummy forward to build weights
    drug_ex([data['drug_data']['atom_features'], data['drug_data']['adjacency_matrix']], training=False)
    cell_ex([data['gexpr_data'], data['copy_number_data'], data['mutation_data']], training=False)
    
    dummy_embed = tf.zeros((1, config.EMBEDDING_DIM))
    contrastive([dummy_embed, dummy_embed], training=False)

    dummy_adj = [tf.sparse.from_dense(tf.eye(num_nodes, dtype=tf.float32))] * config.NUM_RELATIONS
    drug_sim(dummy_adj[0]); cell_sim(dummy_adj[1])
    
    dummy_node_feat = tf.zeros((num_nodes, config.EMBEDDING_DIM))
    rel_ids = tf.range(config.NUM_RELATIONS, dtype=tf.int32)
    rel_embs = rel_emb(rel_ids)

    dgcn1([dummy_node_feat, dummy_adj, rel_embs], training=False)
    dgcn2([dummy_node_feat, dummy_adj, rel_embs], training=False)
    zeros = tf.zeros((1, config.EMBEDDING_DIM))
    distmult([zeros, zeros, zeros], training=False)

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    train_vars = (drug_ex.trainable_variables + cell_ex.trainable_variables +
                  contrastive.trainable_variables + drug_sim.trainable_variables +
                  cell_sim.trainable_variables + dgcn1.trainable_variables +
                  dgcn2.trainable_variables + rel_emb.trainable_variables +
                  distmult.trainable_variables)
    components = {
        'drug_ex': drug_ex, 'cell_ex': cell_ex,
        'contrastive': contrastive, 'drug_sim': drug_sim,
        'cell_sim': cell_sim, 'dgcn1': dgcn1,
        'dgcn2': dgcn2, 'rel_emb': rel_emb,
        'distmult': distmult
    }
    return components, optimizer, train_vars

def train_step(models, optimizer, train_vars, X_tr,res_adj, data, config, num_nodes):
    """Perform one training step and return loss."""
    # Unpack models
    drug_ex = models['drug_ex']; cell_ex = models['cell_ex']
    contrastive = models['contrastive']; drug_sim = models['drug_sim']
    cell_sim = models['cell_sim']; dgcn1 = models['dgcn1']
    dgcn2 = models['dgcn2']; rel_emb = models['rel_emb']
    distmult = models['distmult']

    
    with tf.GradientTape() as tape:
        # Extract features
        drug_feats = drug_ex([data['drug_data']['atom_features'], data['drug_data']['adjacency_matrix']], training=True)
        cell_feats = cell_ex([data['gexpr_data'], data['copy_number_data'], data['mutation_data']], training=True)
        rel_ids = tf.range(config.NUM_RELATIONS, dtype=tf.int32)
        rel_embs = rel_emb(rel_ids)
        node_feats = tf.concat([cell_feats, drug_feats], axis=0)

        # Similarity
        ds, cs = contrastive([drug_feats, cell_feats])
        ds_full = drug_sim(ds); cs_full = cell_sim(cs)

        full_adj = res_adj + [ds_full, cs_full]
        # Negative sampling (use numpy X_tr)
        X_tr_np = X_tr.numpy() if hasattr(X_tr, 'numpy') else X_tr
        drug_sim_triples = sparse_tensor_to_triples(ds_full, rel_id=2)
        cell_sim_triples = sparse_tensor_to_triples(cs_full, rel_id=3)
        all_pos = np.vstack([
            X_tr_np, 
            drug_sim_triples, 
            cell_sim_triples
        ]).astype(np.int32)
        neg = generate_negative_samples(all_pos, num_nodes, config.NUM_RELATIONS, full_adj)

        # DGCN forward
        out1 = dgcn1([node_feats, full_adj + [cs_full, ds_full], rel_embs])
        out2 = dgcn2([out1, full_adj + [cs_full, ds_full], rel_embs])
        final_emb = out1 + out2

        # Prepare scores
        X_tr = tf.cast(all_pos, tf.int32)
        heads_pos, rels_pos, tails_pos = X_tr[:, 0], X_tr[:, 1], X_tr[:, 2]
        heads_neg = np.array([n[0] for n in neg], dtype=np.int32)
        rels_neg  = np.array([n[1] for n in neg], dtype=np.int32)
        tails_neg = np.array([n[2] for n in neg], dtype=np.int32)

        h_pos = tf.gather(final_emb, heads_pos)
        r_pos = tf.gather(rel_embs, rels_pos)
        t_pos = tf.gather(final_emb, tails_pos)

        h_neg = tf.gather(final_emb, heads_neg)
        r_neg = tf.gather(rel_embs, rels_neg)
        t_neg = tf.gather(final_emb, tails_neg)

        h_all = tf.concat([h_pos, h_neg], axis=0)
        r_all = tf.concat([r_pos, r_neg], axis=0)
        t_all = tf.concat([t_pos, t_neg], axis=0)

        pred = distmult([h_all, r_all, t_all])
        y_true = np.concatenate([np.ones(len(X_tr)), np.zeros(len(neg))])

        y_true = np.concatenate([np.ones(len(X_tr)), np.zeros(len(neg))])
        loss_main = compute_loss(y_true, pred)
        loss_contrast = compute_contrastive_loss(drug_feats, cell_feats, contrastive)
        total_loss = (1-config.CONTRAST_WEIGHT) * loss_main + config.CONTRAST_WEIGHT * loss_contrast

    # Compute accuracy
    y_pred = (pred.numpy() > 0.5).astype(int)  # Apply threshold to predictions
    accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy

    # Apply gradients and update model
    grads = tape.gradient(total_loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))
    
    return float(loss_main.numpy()), accuracy  # Return loss and accuracy


def evaluate(models, X_eval, X_tr, data, config, num_nodes):
    """
    Compute evaluation metrics on provided edges (X_eval numpy array of shape [m,3]).
    Returns a dict of metrics.
    """
    base_adj = build_adj_from_edges(
        X_tr, num_nodes, config.NUM_RELATIONS
    )
    # Unpack models
    drug_ex, cell_ex = models['drug_ex'], models['cell_ex']
    contrastive, drug_sim = models['contrastive'], models['drug_sim']
    cell_sim, dgcn1 = models['cell_sim'], models['dgcn1']
    dgcn2, rel_emb = models['dgcn2'], models['rel_emb']
    distmult = models['distmult']

    # Build embeddings once
    drug_feats = drug_ex([data['drug_data']['atom_features'], data['drug_data']['adjacency_matrix']], training=False)
    cell_feats = cell_ex([data['gexpr_data'], data['copy_number_data'], data['mutation_data']], training=False)
    rel_ids = tf.range(config.NUM_RELATIONS, dtype=tf.int32)
    rel_embs = rel_emb(rel_ids)
    node_feats = tf.concat([cell_feats, drug_feats], axis=0)

    ds, cs = contrastive([drug_feats, cell_feats], training=False)
    ds_full = drug_sim(ds); cs_full = cell_sim(cs)

    full_adj = base_adj + [cs_full, ds_full]

    # DGCN forward
    out1 = dgcn1([node_feats, full_adj, rel_embs], training=False)
    out2 = dgcn2([out1, full_adj, rel_embs], training=False)
    final_emb = out1 + out2
    # adj mats to triples
    drug_sim_triples_val = sparse_tensor_to_triples(ds_full, rel_id=2)
    cell_sim_triples_val = sparse_tensor_to_triples(cs_full, rel_id=3)
    all_pos_eval = np.vstack([
        X_eval, 
        drug_sim_triples_val, 
        cell_sim_triples_val
    ]).astype(np.int32)
    # Negative sampling and scores
    neg = generate_negative_samples(all_pos_eval, num_nodes, config.NUM_RELATIONS, full_adj)
    heads_n = np.array([e[0] for e in neg], dtype=np.int32)
    rels_n  = np.array([e[1] for e in neg], dtype=np.int32)
    tails_n = np.array([e[2] for e in neg], dtype=np.int32)
    h_n = tf.gather(final_emb, heads_n)
    r_n = tf.gather(rel_embs, rels_n)
    t_n = tf.gather(final_emb, tails_n)
    neg_scores = distmult([h_n, r_n, t_n]).numpy().flatten()
    # print("-------neg mean/std:", neg_scores.mean(), neg_scores.std())

    # Positive scores
    heads = all_pos_eval[:, 0].astype(np.int32)
    rels  = all_pos_eval[:, 1].astype(np.int32)
    tails = all_pos_eval[:, 2].astype(np.int32)
    h_e = tf.gather(final_emb, heads)
    r_e = tf.gather(rel_embs, rels)
    t_e = tf.gather(final_emb, tails)
    pos_scores = distmult([h_e, r_e, t_e]).numpy().flatten()
    # print("-------pos mean/std:", pos_scores.mean(), pos_scores.std())

    # Combine
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])

    # Calculate the loss
    loss_main = compute_loss(y_true, y_score)  # Compute loss here
    loss_contrast = compute_contrastive_loss(drug_feats, cell_feats, contrastive)
    total_loss = (1-config.CONTRAST_WEIGHT) * loss_main + config.CONTRAST_WEIGHT * loss_contrast
    metrics = compute_metrics(y_true, y_score)
    metrics['loss'] = total_loss
    

    # 假设 pos_scores, neg_scores 分别是正负样本的打分 np.array
    # plt.hist(pos_scores, bins=50, alpha=0.5, label='pos')
    # plt.hist(neg_scores, bins=50, alpha=0.5, label='neg')
    # plt.axvline(0.5000, color='gray', linestyle='--', label='0.50')
    # plt.axvline(0.5022, color='red', linestyle='--', label='F1-th=0.5022')
    # plt.legend(); plt.xlabel('Score'); plt.ylabel('Count')
    # plt.title('Score Distribution Around 0.50'); 
    # plt.show()


    return metrics, y_true, y_score



