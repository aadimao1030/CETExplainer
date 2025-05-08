import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from config import Config
from data_loader import DataLoader
from utils import (setup_environment, build_adj_from_edges, plot_roc_and_aupr, set_random_seed, plot_loss_and_accuracy)
from train_eval import init_dummy_run, train_step, evaluate

def train():
    config = Config()
    set_random_seed(config.SEED)
    setup_environment(config)

    # Load data
    data = DataLoader(config).load_data()
    num_nodes = data['ADJ_MATS'][0].shape[0]
    num_cells = data['gexpr_data'].shape[0]
    components, optimizer, train_vars = init_dummy_run(num_nodes, num_cells, config, data)

    # KFold setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize lists to store results
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(data['X_train']), start=1):
        if fold == 5:
            print(f"\n=== Fold {fold}/5 ===")
            res_adj = build_adj_from_edges(data['X_train'][tr_idx], num_nodes, config.NUM_RELATIONS-2)
            
            # Initialize fold's training and validation losses and accuracies
            fold_train_losses, fold_val_losses = [], []
            fold_train_accuracies, fold_val_accuracies = [], []

            for epoch in range(config.NUM_EPOCHS):
                # Training step
                loss, accuracy = train_step(
                    components, optimizer, train_vars,
                    data['X_train'][tr_idx], res_adj, data, config, num_nodes
                )

                # Record training loss and accuracy
                fold_train_losses.append(loss)
                fold_train_accuracies.append(accuracy)

                print(f"Fold {fold} | Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

                # Validation step
                val_metrics, _, _ = evaluate(
                    components,
                    data['X_train'][val_idx],
                    data['X_train'][tr_idx],
                    data, config, num_nodes
                )
                
                # Record validation loss and accuracy
                fold_val_losses.append(val_metrics['loss'])
                fold_val_accuracies.append(val_metrics['accuracy'])

            # Save the current fold's training and validation losses and accuracies
            train_losses.append(fold_train_losses)
            val_losses.append(fold_val_losses)
            train_accuracies.append(fold_train_accuracies)
            val_accuracies.append(fold_val_accuracies)

            # Evaluate metrics on validation set
            metrics, y_true, y_score = evaluate(
                components,
                data['X_train'][val_idx],
                data['X_train'][tr_idx],
                data, config, num_nodes
            )

            fold_metrics.append({
                'fold': fold,
                'roc_auc': metrics['roc_auc'],
                'aupr': metrics['aupr'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'confusion_matrix': metrics['confusion_matrix']
            })

            # Print fold evaluation results
            print("Evaluation:")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}, AP: {metrics['aupr']:.4f}")
            print(f"Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            print("Confusion Matrix:\n", metrics['confusion_matrix'])

        # plot_roc_and_aupr(y_true, y_score, fold)

    # After all folds, evaluate performance on the test set
    # Assuming test data is available in `data['X_test']`, modify as needed
    test_metrics, y_true_test, y_score_test = evaluate(
        components,
        data['X_test'],  # Test data
        data['X_train'],  # Training data
        data, config, num_nodes
    )

    print("Test Set Evaluation:")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}, AP: {test_metrics['aupr']:.4f}")
    print(f"Acc: {test_metrics['accuracy']:.4f}, Prec: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    print("Confusion Matrix:\n", test_metrics['confusion_matrix'])

    # plot_roc_and_aupr(y_true_test, y_score_test, "test")

    # Optionally, print and/or save the results of each fold's metrics
    print("\nFold-wise Metrics:")
    for fold_metric in fold_metrics:
        print(f"Fold {fold_metric['fold']}:")
        print(f"  ROC AUC: {fold_metric['roc_auc']:.4f}, AP: {fold_metric['aupr']:.4f}")
        print(f"  Accuracy: {fold_metric['accuracy']:.4f}, Precision: {fold_metric['precision']:.4f}, "
              f"Recall: {fold_metric['recall']:.4f}, F1: {fold_metric['f1']:.4f}")

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(fold_metrics).round(4)
    metrics_df.to_csv(f"./result/fold5_metrics.csv", index=False)

    np.save('result/y_true_test2.npy', y_true_test)
    np.save('result/y_score_test2.npy', y_score_test)
    # Plot training and validation loss and accuracy curves
    # plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

if __name__ == '__main__':
    train()
