import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils import (
    load_and_preprocess_mnist,
    create_cnn_model,
    plot_confusion_matrix,
    plot_training_history,
    visualize_clusters
)
import time

def save_model_and_visualizations(model, history, y_pred, features, labels, clusters, x_test, y_test):
    """Sauvegarde le modèle et les visualisations dans des fichiers."""
    try:
        # Création d'un timestamp pour les noms de fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde du modèle
        model_path = os.path.join('models', f'mnist_model_{timestamp}.keras')
        model.save(model_path)
        print(f"\nModèle sauvegardé dans : {model_path}")
        
        # Sauvegarde des visualisations
        vis_path = os.path.join('visualizations', timestamp)
        os.makedirs(vis_path, exist_ok=True)
        
        # Sauvegarde des métriques dans un fichier texte
        metrics_path = os.path.join(vis_path, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"=== Métriques du modèle ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Précision sur l'ensemble de test: {model.evaluate(x_test, y_test)[1]:.4f}\n")
            f.write(f"Score de silhouette: {silhouette_score(features, clusters):.4f}\n")
            f.write(f"\nDistribution des clusters:\n")
            for i in range(10):
                count = np.sum(clusters == i)
                f.write(f"Cluster {i}: {count} échantillons\n")
        print(f"Métriques sauvegardées dans : {metrics_path}")
        
        # 1. Historique d'entraînement
        print("Génération de l'historique d'entraînement...")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('Précision du modèle')
        plt.xlabel('Epoch')
        plt.ylabel('Précision')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Perte du modèle')
        plt.xlabel('Epoch')
        plt.ylabel('Perte')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_path, 'training_history.png'))
        plt.close()
        print("Historique d'entraînement sauvegardé")
        
        # 2. Matrice de confusion
        print("Génération de la matrice de confusion...")
        y_test_pred = model.predict(x_test, batch_size=128)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        plt.figure(figsize=(10, 8))
        cm = tf.math.confusion_matrix(y_test, y_test_pred_classes)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matrice de confusion (Test)')
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, range(10))
        plt.yticks(tick_marks, range(10))
        plt.xlabel('Prédiction')
        plt.ylabel('Vraie étiquette')
        plt.savefig(os.path.join(vis_path, 'confusion_matrix.png'))
        plt.close()
        print("Matrice de confusion sauvegardée")
        
        # 3. Visualisation PCA
        print("Génération de la visualisation PCA...")
        plt.figure(figsize=(12, 8))
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='tab10', alpha=0.6)
        plt.title('Visualisation PCA des clusters')
        plt.colorbar(label='Cluster')
        plt.savefig(os.path.join(vis_path, 'pca_clusters.png'))
        plt.close()
        print("Visualisation PCA sauvegardée")
        
        # 4. Distribution des clusters
        print("Génération de la distribution des clusters...")
        plt.figure(figsize=(12, 6))
        sns.countplot(x=clusters)
        plt.title('Distribution des clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Nombre d\'échantillons')
        plt.savefig(os.path.join(vis_path, 'cluster_distribution.png'))
        plt.close()
        print("Distribution des clusters sauvegardée")
        
        # 5. Visualisation t-SNE (en dernier car c'est le plus long)
        print("Génération de la visualisation t-SNE (cela peut prendre quelques minutes)...")
        plt.figure(figsize=(12, 8))
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        features_tsne = tsne.fit_transform(features)
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=clusters, cmap='tab10', alpha=0.6)
        plt.title('Visualisation t-SNE des clusters')
        plt.colorbar(label='Cluster')
        plt.savefig(os.path.join(vis_path, 'tsne_clusters.png'))
        plt.close()
        print("Visualisation t-SNE sauvegardée")
        
        print(f"\nToutes les visualisations ont été sauvegardées dans : {vis_path}")
        
    except Exception as e:
        print(f"\nErreur lors de la sauvegarde : {str(e)}")
        print("Les fichiers déjà sauvegardés sont conservés.")
        raise

def main():
    # 1. Chargement et prétraitement des données
    print("\n=== Chargement du dataset MNIST ===")
    print("Le dataset sera automatiquement téléchargé depuis le serveur de TensorFlow...")
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    print(f"Dataset chargé en {time.time() - start_time:.2f} secondes")
    print(f"Dimensions des données d'entraînement : {x_train.shape}")
    print(f"Dimensions des données de test : {x_test.shape}")
    print(f"Nombre de classes : {len(np.unique(y_train))}")
    
    # 2. Création et entraînement du modèle
    print("\n=== Entraînement du modèle CNN ===")
    print("Architecture du modèle :")
    model = create_cnn_model()
    model.summary()
    
    print("\nDébut de l'entraînement...")
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"\nEntraînement terminé en {training_time:.2f} secondes")
    
    # 3. Évaluation du modèle
    print("\n=== Évaluation du modèle ===")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Précision sur l'ensemble de test : {test_accuracy:.4f}")
    
    print("\nGénération des prédictions...")
    y_pred = model.predict(x_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # 4. Clustering avec K-means
    print("\n=== Application du clustering K-means ===")
    print("Extraction des features depuis la couche avant-dernière...")
    
    # Création du modèle d'extraction de features
    feature_layer = model.layers[-2]  # La couche Dense(128) avant la dernière
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=feature_layer.output,
        name='feature_extractor'
    )
    
    # Extraction des features
    print("Prédiction des features...")
    features_train = feature_extractor.predict(x_train, batch_size=128)
    features_test = feature_extractor.predict(x_test, batch_size=128)
    print(f"Dimensions des features : {features_train.shape}")
    
    # Application de K-means
    print("\nApplication de K-means...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_train)
    silhouette_avg = silhouette_score(features_train, clusters)
    print(f"Score de silhouette moyen : {silhouette_avg:.4f}")
    
    # 5. Sauvegarde du modèle et des visualisations
    print("\n=== Sauvegarde des résultats ===")
    save_model_and_visualizations(
        model=model,
        history=history.history,
        y_pred=y_pred_classes,
        features=features_train,
        labels=y_train,
        clusters=clusters,
        x_test=x_test,
        y_test=y_test
    )

if __name__ == "__main__":
    main() 