import tensorflow as tf
import numpy as np
from prediction.code import utils
import random as rn
from prediction.code import DGCN
import pandas as pd
import os
import time
from tqdm import tqdm
import datetime

def get_neighbors(data_subset, node_idx):
    # Get neighbors for a given node index
    head_neighbors = tf.boolean_mask(data_subset, data_subset[:, 0] == node_idx)
    tail_neighbors = tf.boolean_mask(data_subset, data_subset[:, 2] == node_idx)
    neighbors = tf.concat([head_neighbors, tail_neighbors], axis=0)
    return neighbors

def get_computation_graph(head, rel, tail, data, num_relations):
    neighbors_head = get_neighbors(data, head)
    neighbors_tail = get_neighbors(data, tail)
    all_neighbors = tf.concat([neighbors_head, neighbors_tail], axis=0)
    return all_neighbors

def structure_loss(masked_adj_mats, target_ratios,epoch):
    # Initialize a list to count relations
    relation_counts = [
        tf.sparse.reduce_sum(adj) if isinstance(adj, tf.SparseTensor) else tf.reduce_sum(adj)
        for adj in masked_adj_mats
    ]
    total_count = tf.reduce_sum(relation_counts)

    actual_ratios = tf.stack([count / total_count for count in relation_counts])
    if (epoch+1)%10==0:
        print(epoch)
        print(actual_ratios)

    loss = tf.reduce_mean(tf.square(target_ratios - actual_ratios))
    return loss

def cetexplainer_step(head, rel, tail, num_entities, num_relations,ADJACENCY_DATA, init_value,
                      masks, optimizer,TARGET_RATIOS,THRESHOLD,writer):
    # Get computation graph
    comp_graph = get_computation_graph(head, rel, tail, ADJACENCY_DATA, num_relations)
    # Get adjacency matrices
    adj_mats = utils.get_adj_mats(comp_graph, num_entities, num_relations)
    for epoch in range(NUM_EPOCHS):

        with tf.GradientTape() as tape:

            tape.watch(masks)
            masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(num_relations)]

            before_pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                adj_mats
            ])
            pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                masked_adjs
            ])
            # Get predictions before and after masking
            struct_loss = structure_loss(masked_adjs, TARGET_RATIOS,epoch)

            pred_loss = - before_pred * tf.math.log(pred + 0.00001)
            loss = (pred_loss + struct_loss)/2.

            # Print loss every 10 epochs
            if (epoch+1) % 10 == 0:
                print(f"current loss {loss}")

            # Record loss with TensorBoard
            scalar_loss = tf.squeeze(loss)
            with writer.as_default():
                tf.summary.scalar('all_loss', scalar_loss.numpy(), step=epoch)

                writer.flush()

        # Apply gradients to update masks
        grads = tape.gradient(loss, masks)
        optimizer.apply_gradients(zip(grads, masks))

    # Generate explanations
    current_pred = []
    current_scores = []

    for i in range(num_relations):
        # Generate explanations
        mask_i = adj_mats[i] * tf.sigmoid(masks[i])
        # Get indices not masked
        mask_idx = mask_i.values > THRESHOLD
        non_masked_indices = tf.gather(mask_i.indices[mask_idx], [1, 2], axis=1)

        # If there are non-masked indices, add them to the prediction list
        if tf.reduce_sum(non_masked_indices) != 0:
            # rel_indices = tf.cast(tf.ones((non_masked_indices.shape[0], 1)) * i, tf.int64)
            if non_masked_indices.shape[0] is not None and non_masked_indices.shape[0] > 0:
                rel_indices = tf.cast(tf.ones((non_masked_indices.shape[0], 1)) * i, tf.int64)
            else:
                print(non_masked_indices)
            triple = tf.concat([non_masked_indices, rel_indices], axis=1)
            triple = tf.gather(triple, [0, 2, 1], axis=1)
            score_array = mask_i.values[mask_idx]
            current_pred.append(triple)
            current_scores.append(score_array)
    # Concatenate scores and get top scores
    current_scores = tf.concat([array for array in current_scores], axis=0)
    top_k_scores = tf.argsort(current_scores, direction='DESCENDING')[0:10]
    # Reshape predictions and select top scored predictions
    pred_exp = tf.reshape(tf.concat([array for array in current_pred], axis=0), (-1, 3))
    pred_exp = tf.gather(pred_exp, top_k_scores, axis=0)
    # Reset all masks to initial value
    for mask in masks:
        mask.assign(value=init_value)
    return pred_exp


if __name__ == '__main__':
    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    NUM_EPOCHS = 50
    EMBEDDING_DIM = 64
    LEARNING_RATE = 0.01

    NUM_ENTITIES = 634
    NUM_RELATIONS = 4
    OUTPUT_DIM = EMBEDDING_DIM
    THRESHOLD = 0.15

    TARGET_RATIOS = tf.constant([0.1, 0.4, 0.4, 0.1], dtype=tf.float32)
    fold = 4

    start_time = time.time()



    log_dir = "logs1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir)

    train2idx = pd.read_csv(fr'../prediction/data/split_data/mode0_fold{fold}_X_train.csv', header=0, dtype=int).values
    test2idx = pd.read_csv(fr'data/test_filtered_fold{fold}.csv', header=0, dtype=int).values
    ALL_INDICES = tf.reshape(tf.range(0, NUM_ENTITIES, 1, dtype=tf.int64), (1, -1))
    all_feature_matrix = pd.read_csv(r"../prediction/data/node_representation/x_all_64_epoch5000.csv", header=0)

    model = DGCN.get_DGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED,
        all_feature_matrix=all_feature_matrix,
        mode=0,
        fold=fold
    )

    model.load_weights(os.path.join(
        f'../prediction/data/weights/mode0_fold{fold}_epoch5000_learnRate0.001_batchsize256_embdim64_layer2.h5'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    init_value = tf.random.normal(
        (1, NUM_ENTITIES, NUM_ENTITIES),
        mean=0,
        stddev=1,
        dtype=tf.float32,
        seed=SEED)
    masks = [tf.Variable(
        initial_value=init_value,
        name='mask_' + str(i),
        trainable=True) for i in range(NUM_RELATIONS)]

    ADJACENCY_DATA = tf.concat([train2idx, test2idx], axis=0)

    del train2idx

    tf_data = tf.data.Dataset.from_tensor_slices((test2idx[:, 0], test2idx[:, 1], test2idx[:, 2])).batch(1)

    best_preds = []
    for head, rel, tail in tf_data:
        # print(head, rel, tail)
        print("%d %d %d" % (head, rel, tail))
        current_preds = cetexplainer_step(head, rel, tail, NUM_ENTITIES, NUM_RELATIONS, ADJACENCY_DATA,
                                                init_value, masks, optimizer, TARGET_RATIOS,THRESHOLD,writer)
        best_preds.append(current_preds)

    best_preds = [array.numpy() for array in best_preds]

    out_preds = []
    for i in tqdm(range(len(best_preds))):
        preds_i = utils.idx2array(best_preds[i])
        out_preds.append(preds_i)
    out_preds = np.array(out_preds, dtype=object)

    current_time = int(time.time())
    np.savez(f'data/CETExplainer_preds_layer2_epoch5000.npz', preds=out_preds)

    duration = time.time()-start_time

    print(f'Time: {duration/3600}h')


    print(f'Num epochs: {NUM_EPOCHS}')
    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f'Learning rate: {LEARNING_RATE}')
    print(f'Threshold {THRESHOLD}')
    print('Done.')
