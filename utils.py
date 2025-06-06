import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os
import pickle
import pandas as pd

def ensure_data_directories():
    """Crée les dossiers nécessaires s'ils n'existent pas."""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

def load_and_preprocess_mnist():
    """Charge et prétraite le dataset MNIST depuis les fichiers CSV locaux."""
    ensure_data_directories()
    
    # Chemins des fichiers
    train_path = 'data/raw/mnist_train.csv'
    test_path = 'data/raw/mnist_test.csv'
    processed_data_path = 'data/processed/mnist_processed.pkl'
    
    # Vérifier si les données prétraitées existent déjà
    if os.path.exists(processed_data_path):
        print("Chargement des données prétraitées depuis le cache...")
        with open(processed_data_path, 'rb') as f:
            return pickle.load(f)
    
    print("Chargement des données depuis les fichiers CSV...")
    # Chargement des données d'entraînement
    train_data = pd.read_csv(train_path)
    x_train = train_data.iloc[:, 1:].values  # Toutes les colonnes sauf la première (étiquettes)
    y_train = train_data.iloc[:, 0].values   # Première colonne (étiquettes)
    
    # Chargement des données de test
    test_data = pd.read_csv(test_path)
    x_test = test_data.iloc[:, 1:].values    # Toutes les colonnes sauf la première (étiquettes)
    y_test = test_data.iloc[:, 0].values     # Première colonne (étiquettes)
    
    print(f"Dimensions des données d'entraînement : {x_train.shape}")
    print(f"Dimensions des données de test : {x_test.shape}")
    
    # Prétraitement
    print("Prétraitement des données...")
    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Les données sont déjà en format aplati (784 features), pas besoin de reshape
    
    # Sauvegarder les données prétraitées
    print("Sauvegarde des données prétraitées...")
    processed_data = ((x_train, y_train), (x_test, y_test))
    with open(processed_data_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    return processed_data

def create_cnn_model():
    """Crée un modèle CNN pour la classification MNIST en utilisant l'API fonctionnelle."""
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, title='Matrice de Confusion'):
    """Affiche la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Prédiction')
    plt.show()

def visualize_clusters(data, labels, method='pca', title='Visualisation des Clusters'):
    """Visualise les clusters en 2D avec PCA ou t-SNE."""
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA - ' + title
    else:  # t-sne
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE - ' + title
    
    # Réduction de dimensionnalité
    reduced_data = reducer.fit_transform(data)
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')
    plt.show()

def plot_training_history(history):
    """Affiche l'historique d'entraînement du modèle."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Précision du modèle')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Précision')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Perte du modèle')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Perte')
    ax2.legend()
    
    plt.tight_layout()
    plt.show() 