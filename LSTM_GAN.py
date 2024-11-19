import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
import re

# Data

# Fonction pour charger et combiner les séries temporelles en une série multivariée
def load_multidimensional_series(file_paths):
    data_list = []
    
    for file_path in file_paths:
        # Charger les données depuis le fichier .mat
        mat_data = scipy.io.loadmat(file_path)
        
        # Afficher les clés disponibles dans le fichier .mat
        available_keys = list(mat_data.keys())
        print(f"Clés disponibles dans {file_path}: {available_keys}")
        
        combined_series = []
        
        # Rechercher dynamiquement les clés basées sur le schéma "X###_DE_time", "X###_FE_time", "X###_BA_time"
        de_time_key = next((key for key in available_keys if re.match(r"X\d+_DE_time", key)), None)
        fe_time_key = next((key for key in available_keys if re.match(r"X\d+_FE_time", key)), None)
        ba_time_key = next((key for key in available_keys if re.match(r"X\d+_BA_time", key)), None)

        # Extraire les données si les clés existent
        for key in [de_time_key, fe_time_key, ba_time_key]:
            if key:
                print(f"Utilisation de la clé {key} dans {file_path}")
                data = mat_data[key].flatten()  # Aplatir si nécessaire
                combined_series.append(data)
            else:
                print(f"Clé correspondante non trouvée dans {file_path}")
        
        if combined_series:
            # Si on a trouvé des séries valides, les empiler
            combined_series = np.stack(combined_series, axis=1)  # Stack pour créer une matrice [n_samples, n_dimensions]
            data_list.append(combined_series)
        else:
            print(f"Aucune clé valide trouvée dans {file_path}")
    
    if data_list:
        # Combiner les différentes séries dans une matrice 3D [n_samples, n_dimensions, n_files]
        multivariate_series = np.concatenate(data_list, axis=0)  # Si vous voulez les concaténer sur la dimension des échantillons
        return multivariate_series
    else:
        print("Aucune donnée valide trouvée dans les fichiers fournis.")
        return None

# Fonction pour visualiser la série temporelle multivariée
def plot_multivariate_series(series, dimension_names=None, start=0, end=1000):
    n_dimensions = series.shape[1]
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_dimensions):
        plt.subplot(n_dimensions, 1, i + 1)
        plt.plot(series[start:end, i])
        plt.title(dimension_names[i] if dimension_names else f"Dimension {i + 1}")
        plt.xlabel("Échantillon")
        plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()

# Chemins vers les fichiers .mat associés à un défaut spécifique (exemple : Ball/0007)
base_dir = "./CRWU/12k Drive End Bearing Fault Data/Ball/0007"
file_names = ["B007_0.mat", "B007_1.mat", "B007_2.mat", "B007_3.mat"]

# Obtenir les chemins complets des fichiers
file_paths = [os.path.join(base_dir, file_name) for file_name in file_names]

# Charger et combiner les séries
multivariate_series = load_multidimensional_series(file_paths)

# GAN

data = multivariate_series

# Découper les données en séquences temporelles
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# longueur de la séquence à générer
sequence_length = 100  
data = create_sequences(data, sequence_length)

# Vérif forme données
print(f"Nouvelle forme de data: {data.shape}")  # (n_sequences, sequence_length, 3)

# La forme est maintenant correcte pour l'entraînement LSTM
n_sequences, sequence_length, n_dimensions = data.shape

# générateur
def build_generator(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(3)))  # 3 sorties pour DE, FE, BA
    return model

# Définir le discriminateur
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(1, activation='sigmoid'))  # Classification binaire (vrai/faux)
    return model

# Construire le GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(sequence_length, 3))  # 3 séries temporelles
    generated_series = generator(gan_input)
    gan_output = discriminator(generated_series)
    
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# Boucle d'entraînement
def train_gan(generator, discriminator, gan, data, epochs=1000, batch_size=32):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Générer du bruit aléatoire comme entrée pour le générateur
        noise = np.random.normal(0, 1, (half_batch, sequence_length, 3))

        # Générer des données synthétiques (séries temporelles)
        generated_data = generator.predict(noise)
        
        # Sélectionner un demi-lot aléatoire de vraies données
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        
        # Entraîner le discriminateur
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Entraîner le générateur (via le GAN)
        noise = np.random.normal(0, 1, (batch_size, sequence_length, 3))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

# Construire et compiler les modèles
generator = build_generator(input_shape=(sequence_length, 3))
discriminator = build_discriminator(input_shape=(sequence_length, 3))
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

gan = build_gan(generator, discriminator)

# Entraîner le GAN
train_gan(generator, discriminator, gan, data)

# Générer des échantillons de séries temporelles futures
noise = np.random.normal(0, 1, (1, sequence_length, 3))
generated_series = generator.predict(noise)

# Visualiser les séries temporelles générées
plt.figure(figsize=(12, 6))
plt.plot(generated_series[0][:, 0], label='Generated Drive End (DE)')
plt.plot(generated_series[0][:, 1], label='Generated Fan End (FE)')
plt.plot(generated_series[0][:, 2], label='Generated Base Acceleration (BA)')
plt.title('Generated Time Series using GAN')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()