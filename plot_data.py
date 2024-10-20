import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
import re

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

# Visualiser la série multivariée si elle est chargée correctement
if multivariate_series is not None:
    # Nom des dimensions (facultatif)
    dimension_names = ["Drive End (DE)", "Fan End (FE)", "Base Acceleration (BA)"]
    
    # Visualiser les 1000 premiers échantillons (ou ajuster la plage avec start et end)
    plot_multivariate_series(multivariate_series, dimension_names, start=0, end=1000)