import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


# Modèle MLP
class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(input_size, 500),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(500, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

class VGGModel(nn.Module):
    def __init__(self, num_classes):
        super(VGGModel, self).__init__()
        self.base_model = models.vgg16(pretrained=False)
        self.base_model.classifier[6] = nn.Sequential(
            nn.Linear(4096, num_classes),
            nn.Dropout(0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Vérification de la forme de l'entrée et ajout de dimensions nécessaires
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(2)  # (batch_size, 1024, 1) -> (batch_size, 1024)
        
        # Redimensionner pour ajouter la dimension spatiale de 1 (batch_size, 1024, 1) -> (batch_size, 1, 1024, 1)
        x = x.unsqueeze(1)  # (batch_size, 1024) -> (batch_size, 1, 1024, 1)
        
        # Appliquer l'interpolation pour redimensionner la taille spatiale à (224, 224)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Répliquer le canal pour obtenir 3 canaux
        x = x.expand(-1, 3, -1, -1)  # (batch_size, 1, 224, 224) -> (batch_size, 3, 224, 224)

        return self.base_model(x)


class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Réorganiser la forme en (batch_size, 1, 1024)
        x = torch.nn.functional.interpolate(x, size=(224, 224))  # Redimensionner à 224x224 pour ResNet
        x = x.expand(-1, 3, -1, -1)  # Dupliquer pour 3 canaux
        return self.base_model(x)

class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(1, 100, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(100, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class BLSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(BLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(1, 100, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(200, 100, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(200, num_classes),  # 100 pour chaque direction
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        lstm_out1, _ = self.lstm1(x)
        _, (hn2, _) = self.lstm2(lstm_out1)
        hn = torch.cat((hn2[-2], hn2[-1]), dim=1)
        return self.fc(hn)


class LSTMFCNModel(nn.Module):
    def __init__(self, num_classes):
        super(LSTMFCNModel, self).__init__()
        # LSTM qui génère une sortie de taille (batch_size, seq_len, 128)
        self.lstm = nn.LSTM(1, 128, batch_first=True)
        self.dropout = nn.Dropout(0.8)

        # Couches convolutionnelles (attendent 128 canaux en entrée)
        self.conv1 = nn.Conv1d(1, 128, kernel_size=8, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 265, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv1d(265, 128, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Couches fully connected
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Passage dans le LSTM
        lstm_out, (hn, _) = self.lstm(x)  # lstm_out: (batch_size, seq_len, 128)
        lstm_out = self.dropout(lstm_out)

        conv_input = x.permute(0, 2, 1)  # Convertir en (batch_size, 1, seq_len) pour Conv1d
        x = torch.relu(self.conv1(conv_input))  # (batch_size, 265, seq_len)
        x = torch.relu(self.conv2(x))           # (batch_size, 128, seq_len)
        x = torch.relu(self.conv3(x))           # (batch_size, 128, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)  # Moyenne globale sur la dimension de la séquence (batch_size, 128)


        # Combine la sortie du LSTM avec celle des convolutions
        combined = torch.cat((lstm_out[:, -1, :], x), dim=1)  # Concaténation de la dernière sortie du LSTM et des convolutions
        return self.fc(combined)  # Classification finale


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
                
        optimizer.zero_grad()
        outputs = model(data)
        _, true_labels = labels.max(1)
        
        loss = criterion(outputs, true_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # Convertir les labels one-hot en indices
        # Obtenir les indices des prédictions
        _, preds = outputs.max(1)
        
        # Calcul de l'exactitude
        correct_preds += (preds == true_labels).sum().item()
        total_preds += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds
    return epoch_loss, epoch_accuracy

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_preds = []  # Liste pour collecter toutes les prédictions
    all_labels = []  # Liste pour collecter toutes les véritables étiquettes
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            _, true_labels = labels.max(1)
            
            loss = criterion(outputs, true_labels)
            running_loss += loss.item()
            
            # Obtenir les indices des prédictions
            _, preds = outputs.max(1)
            
            all_preds.append(preds.cpu().numpy())  
            
            # Calcul de l'exactitude
            correct_preds += (preds == true_labels).sum().item()
            total_preds += labels.size(0)
    
    # Convertir les listes de prédictions et d'étiquettes en tableaux numpy
    all_preds = np.concatenate(all_preds)
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_preds / total_preds
    
    return val_loss, val_accuracy, all_preds    
