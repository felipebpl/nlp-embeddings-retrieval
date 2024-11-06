import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Carregar os embeddings pré-treinados e ajustados
pretrained_embeddings = torch.load('embeddings/document_embeddings.pth')
trained_embeddings = torch.load('embeddings/trained_embeddings.pth')

# Função para aplicar TSNE e visualizar
def plot_tsne(embeddings, title="Embeddings"):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=50, alpha=0.7, edgecolors="k", linewidth=0.5)
    plt.title(title, fontsize=16, fontweight="bold", color="navy")
    plt.xlabel("TSNE Dimension 1", fontsize=14)
    plt.ylabel("TSNE Dimension 2", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    output_dir = 'step2_output'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/{title.replace(' ', '_').lower()}.png"
    plt.savefig(file_name)
    
    plt.show()
    plt.close()
    
    print(f"✔️ Plot for '{title}' saved successfully as '{file_name}'\n")
    
    return embeddings_2d

# Gerar gráficos TSNE para embeddings pré-treinados e ajustados
print("🔍 Generating TSNE plots for embeddings...\n")
pretrained_2d = plot_tsne(pretrained_embeddings, title="Pre-trained Embeddings")
trained_2d = plot_tsne(trained_embeddings, title="Trained Embeddings")

# Função para calcular métricas de avaliação de clustering
def evaluate_clustering(embeddings, description):
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    
    print(f"📊 Clustering Evaluation for {description}:\n")
    print(f"   - Silhouette Score (higher is better): {silhouette:.4f}")
    print(f"   - Calinski-Harabasz Index (higher is better): {calinski_harabasz:.4f}")
    print(f"   - Davies-Bouldin Index (lower is better): {davies_bouldin:.4f}")
    print("-" * 50 + "\n")

# Avaliar métricas para os embeddings
print("🔍 Evaluating clustering metrics for embeddings...\n")
evaluate_clustering(pretrained_2d, description="Pre-trained Embeddings")
evaluate_clustering(trained_2d, description="Trained Embeddings")