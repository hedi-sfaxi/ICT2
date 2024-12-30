import torch

from sklearn.metrics import accuracy_score
import torch.distributions as dist
import numpy as np
import torch.optim as optim
from evaluation_models import VGGModel, MLPModel, ResNetModel, LSTMFCNModel, LSTMModel, BLSTMModel, train_model, evaluate_model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

def kl_divergence(real_data, generated_data, num_bins = 64):

    """
    Calcule la divergence KL entre les données réelles et générées en utilisant torch.nn.functional.kl_div.
    """
    # Flatten les données
    real_data = real_data.detach().cpu().flatten()
    generated_data = generated_data.detach().cpu().flatten()
    

    # Calculer les histogrammes normalisés
    real_hist, bin_edges = torch.histogram(real_data, bins=num_bins, density=True)
    generated_hist, _ = torch.histogram(generated_data, bins=bin_edges, density=True)

    # Ajouter un petit offset pour éviter les zéros
    epsilon = 1e-10
    real_hist = real_hist + epsilon
    generated_hist = generated_hist + epsilon

    # Normaliser les histogrammes
    real_hist /= real_hist.sum()
    generated_hist /= generated_hist.sum()

    # Convertir les probabilités en log-probabilités
    real_log_hist = torch.log(real_hist)

    # Calculer la divergence KL
    kl_div = F.kl_div(real_log_hist, generated_hist, reduction="batchmean")
    return kl_div.item()


def visualization(real_data, generated_data):
    
    real_data = np.squeeze(real_data)
    generated_data = np.squeeze(generated_data)

    pca = PCA(n_components=2)
    real_data_pca = pca.fit_transform(real_data)
    generated_data_pca = pca.transform(generated_data)
    
    # Appliquer t-SNE sur les données réelles
    tsne = TSNE(n_components=2, perplexity = 40, n_iter = 300)
    all_data = np.concatenate((real_data, generated_data), axis=0)
    data_tsne = tsne.fit_transform(all_data)
    
    # Création des graphiques
    plt.figure(figsize=(15, 6))
    
    # Plot PCA
    plt.subplot(1,2,1)
    plt.scatter(real_data_pca[:, 0], real_data_pca[:, 1], color='blue', label='Real Data', alpha=0.5)
    plt.scatter(generated_data_pca[:, 0], generated_data_pca[:, 1], color='red', label='Generated Data', alpha=0.5)
    plt.title('PCA')
    plt.xlabel('x_PCA')
    plt.ylabel('y_PCA ')
    plt.legend()
    
    # Plot t-SNE
    plt.subplot(1,2,2)
    plt.scatter(data_tsne[:real_data.shape[0], 0], data_tsne[:real_data.shape[0], 1], color='blue', label='Real Data', alpha=0.5)
    plt.scatter(data_tsne[real_data.shape[0]:, 0], data_tsne[real_data.shape[0]:, 1], color='red', label='Generated Data', alpha=0.5)
    plt.title('t-SNE')
    plt.xlabel('x_t-SNE')
    plt.ylabel('y_t-SNE ')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def test_set_evaluation(model, train_data, test_data, num_epochs=10, batch_size=32, learning_rate=1e-3, device='cuda'):
    """
    Evaluate a model on a provided test set.

    Parameters:
        model_class: The model class to instantiate.
        train_data: The training dataset.
        test_data: The test dataset.
        labels: The labels of the training data.
        num_epochs: Number of epochs for training.
        batch_size: Batch size for DataLoader.
        learning_rate: Learning rate for the optimizer.
        device: Device to use ('cuda' or 'cpu').

    Returns:
        A dictionary containing train loss, train accuracy, test loss, and test accuracy.
    """
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = model.to(device)
    # Loss and optimizer
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_test_accuracy = 0
    best_test_loss = 0
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")
        
        test_loss, test_accuracy, preds = evaluate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
        if best_test_accuracy < test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_loss = test_loss
            best_preds = preds

    print(f"Best Test Loss: {best_test_loss:.4f} - Best Test Accuracy: {best_test_accuracy:.4f}")
    # Return results
    return {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': best_test_loss,
        'test_accuracy': best_test_accuracy,
        'preds': best_preds
    }

# Example usage for different models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_evaluation(train_dataset, test_dataset, input_size, num_classes, num_epochs=10, batch_size=64, learning_rate=1e-3, device='cuda'):
    # Define your models
    models_dict = {
        'MLP': MLPModel(input_size=input_size, num_classes=num_classes),
        # 'VGG': VGGModel(num_classes=num_classes),  
        # 'ResNet': ResNetModel(num_classes=num_classes),  
        'LSTM': LSTMModel(num_classes=num_classes),  
        'BLSTM': BLSTMModel(num_classes=num_classes),  
        'LSTM-FCN': LSTMFCNModel( num_classes=num_classes), 
    }
    
    results = {}

    # Train and evaluate the models using cross-validation
    for model_name, model_class in models_dict.items():
        print(f"\nEvaluating {model_name} model:")
        results[model_name] = test_set_evaluation(model_class, train_dataset, test_dataset, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, device=device)
    return results