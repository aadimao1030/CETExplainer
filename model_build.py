from tensorflow.keras.layers import Embedding
from layers import (
    DGCNLayer,
    ContrastiveLearning,
    CellFeatureExtractor,
    DrugFeatureExtractor,
    SimilarityEmbeddingLayer
)


def build_model(num_nodes, num_cells, config):
    """
    Instantiate all components for the enhanced DGCN model.

    Returns:
        drug_feature_extractor: DrugFeatureExtractor
        cell_feature_extractor: CellFeatureExtractor
        contrastive: ContrastiveLearning
        drug_sim_embedder: SimilarityEmbeddingLayer
        cell_sim_embedder: SimilarityEmbeddingLayer
        dgcn_layer1: EnhancedDGCNLayer
        dgcn_layer2: EnhancedDGCNLayer
        relation_embedding_layer: Embedding
    """
    # === Feature extraction ===
    drug_feature_extractor = DrugFeatureExtractor(
        output_dim=config.EMBEDDING_DIM,
        dropout_rate=config.DROPOUT_RATE
    )
    cell_feature_extractor = CellFeatureExtractor(
        output_dim=config.EMBEDDING_DIM,
        dropout_rate=config.DROPOUT_RATE
    )

    # === Contrastive learning ===
    contrastive = ContrastiveLearning(
        temperature=config.TEMPERATURE,
        projection_dim=config.EMBEDDING_DIM
    )

    # === Similarity embedding ===
    drug_sim_embedder = SimilarityEmbeddingLayer(
        start_idx=num_cells,
        full_size=num_nodes
    )
    cell_sim_embedder = SimilarityEmbeddingLayer(
        start_idx=0,
        full_size=num_nodes
    )

    # === Relation embeddings ===
    relation_embedding_layer = Embedding(
        input_dim=config.NUM_RELATIONS,
        output_dim=config.EMBEDDING_DIM,
        name='relation_embedding'
    )

    # === DGCN layers ===
    dgcn_layer1 = DGCNLayer(
        embedding_dim=config.EMBEDDING_DIM,
        num_relations=config.NUM_RELATIONS,
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=config.L2_REG,
        res_mats_dim=(num_nodes, num_nodes)
    )
    dgcn_layer2 = DGCNLayer(
        embedding_dim=config.EMBEDDING_DIM,
        num_relations=config.NUM_RELATIONS,
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=config.L2_REG,
        res_mats_dim=(num_nodes, num_nodes)
    )

    return (
        drug_feature_extractor,
        cell_feature_extractor,
        contrastive,
        drug_sim_embedder,
        cell_sim_embedder,
        dgcn_layer1,
        dgcn_layer2,
        relation_embedding_layer
    )
