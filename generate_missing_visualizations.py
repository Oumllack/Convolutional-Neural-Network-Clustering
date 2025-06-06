import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_preprocess_mnist, create_cnn_model

# Paths
VIS_PATH = os.path.join('visualizations', '20250607_053032')
MODEL_PATH = os.path.join('models', 'mnist_model_20250607_053032.keras')

# 1. Load data and model
(x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Feature extraction
feature_layer = model.layers[-2]
feature_extractor = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)
features_train = feature_extractor.predict(x_train, batch_size=128)

# 3. K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_train)

# 4. Cluster Distribution Plot
plt.figure(figsize=(12, 6))
sns.countplot(x=clusters)
plt.title('Cluster Distribution')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.savefig(os.path.join(VIS_PATH, 'cluster_distribution.png'))
plt.close()
print('Cluster distribution plot saved.')

# 5. t-SNE Visualization
print('Generating t-SNE visualization (this may take a few minutes)...')
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
features_tsne = tsne.fit_transform(features_train)
plt.figure(figsize=(12, 8))
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=clusters, cmap='tab10', alpha=0.6)
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster Label')
plt.savefig(os.path.join(VIS_PATH, 'tsne_clusters.png'))
plt.close()
print('t-SNE cluster plot saved.') 