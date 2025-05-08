import os

class Config:
    # ========== Data Paths ==========
    DATA_ROOT = "./data"
    SMILES_PATH = os.path.join(DATA_ROOT, "isosmiles.csv")
    MUTATION_PATH = os.path.join(DATA_ROOT, "mutation.csv")
    GEXPR_PATH = os.path.join(DATA_ROOT, "expression.csv")
    COPY_NUMBER_PATH = os.path.join(DATA_ROOT, "copy_number.csv")
    RESPONSE_PATH = os.path.join(DATA_ROOT, "GDSC.csv")
    
    # ========== Random Seed & Device ==========
    SEED = 89

    # ========== Training Parameters ==========
    NUM_EPOCHS = 200

    LEARNING_RATE = 0.001
    
    # ========== Number of Relations ==========
    NUM_RELATIONS = 4
    
    # ========== Embedding Dimensions ==========
    EMBEDDING_DIM = 128
    TEMPERATURE = 0.1
    CONTRAST_WEIGHT = 0.1   # Contrast loss weight
    DROPOUT_RATE = 0.1

    # ========== Graph Convolution & Regularization ==========
    L2_REG = 0.01
