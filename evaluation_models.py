import torch
import torch.nn as nn
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

class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_BiLSTM, self).__init__()
        
        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM Layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(64 * 2, num_classes)
    
    def forward(self, x):
        
        x = x.permute(0,2,1) # (batch_size, channels, seq_len)
        # CNN forward
        x = self.pool(torch.relu(self.conv1(x)))
        
        # Prepare input for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        
        # LSTM forward
        x, _ = self.lstm(x)
        
        # Output layer
        x = self.fc(x[:, -1, :])  # Get the output of the last timestep
        
        return x
    
class WDCNN(nn.Module):
    def __init__(self, num_classes):
        super(WDCNN, self).__init__()

        # Première couche CNN (Wide)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=1, padding=32)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Autres couches CNN (Deep)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Couches fully connected
        self.fc1 = nn.Linear(64 * (1024 // 2), 512)  # assuming seq_len=1024 and max pool reduces seq_len by 2
        self.fc2 = nn.Linear(512, num_classes)  # For binary classification (adjust output size as needed)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through first convolution and pooling layer4
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Pass through subsequent convolutional layers
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, model_dim=64, n_encoders=4, n_decoders=4, n_heads=4, hidden_dim=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, model_dim)
        
        # Transformer Encoder and Decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoders)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_decoders)
        
        # Output Layer
        self.fc_out = nn.Linear(model_dim, num_classes)
    
    def forward(self, src):
        # Embedding de l'entrée (transformer la série temporelle en embeddings)
        src_emb = self.embedding(src)
        
        # Encoder forward pass
        memory = self.transformer_encoder(src_emb)
        
        # Prendre la sortie de l'encodeur et faire une réduction sur la séquence (par exemple, la moyenne)
        pooled_output = memory.mean(dim=1)  # Vous pouvez aussi essayer d'utiliser le dernier état
        
        # Prédiction de classe
        output = self.fc_out(pooled_output)
        
        return output

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
                
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = outputs.max(1)
        
        # Calcul de l'exactitude
        correct_preds += (preds == labels).sum().item()
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
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Obtenir les indices des prédictions
            _, preds = outputs.max(1)
            
            all_preds.append(preds.cpu().numpy())  
            
            # Calcul de l'exactitude
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
    
    # Convertir les listes de prédictions et d'étiquettes en tableaux numpy
    all_preds = np.concatenate(all_preds)
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_preds / total_preds
    
    return val_loss, val_accuracy, all_preds    
