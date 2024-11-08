# Vector-Based Search System Using BERT and Autoencoder

This project implements a vector-based search system using BERT embeddings and an autoencoder to improve semantic relevance in document search. The goal is to create embeddings for each document and query, then use cosine similarity to retrieve the most relevant documents. This project was developed in multiple steps, as detailed below.

The official submission document with detailed discussions, analyses, and results as per the APS 2 assignment requirements is available as **APS2.pdf**.


---

## Table of Contents
1. [Background](#background)
2. [Project Overview](#project-overview)
3. [Steps](#steps)
   - [Step 1: Find Embeddings](#step-1-find-embeddings)
   - [Step 2: Visualize Embeddings](#step-2-visualize-embeddings)
   - [Step 3: Test the Search System](#step-3-test-the-search-system)
4. [Results](#results)
5. [Usage](#usage)

---

## Background

The project is inspired by the method presented in:
> Venkatesh Sharma, K., Ayiluri, P.R., Betala, R. et al. *Enhancing query relevance: leveraging SBERT and cosine similarity for optimal information retrieval*. Int J Speech Technol 27, 753–763 (2024).

The main goal of this project is to improve the semantic relevance of search results by representing both documents and queries in an embedding space using BERT. The embeddings are further refined through an autoencoder, enabling enhanced contextual accuracy in information retrieval.

---

## Project Overview

In this system:
1. **BERT** is used to generate initial embeddings for each document, providing a rich semantic foundation.
2. **Semantic Autoencoder** is applied to adjust these embeddings, compressing them into a lower-dimensional space (128 dimensions) while preserving essential contextual features specific to the dataset.
3. **Cosine Similarity** is used to calculate the relevance between the query embedding and document embeddings, efficiently retrieving the most contextually appropriate documents.

---

## Steps

### Step 1: Generate Embeddings

In this initial step, embeddings were generated for each document in the dataset using BERT, followed by dimensionality reduction through an autoencoder.

1. **Dataset**: The dataset includes blog posts by Paul Graham, which feature recurring themes and nuanced language.
2. **BERT Model**: We utilized the `bert-base-nli-mean-tokens` model from the `sentence-transformers` library to generate the initial document embeddings (768-dimensional vectors).
3. **Semantic Autoencoder**: A multi-layer autoencoder was implemented with the following structure:
   - **Input Layer**: 768 dimensions (matching the BERT output size)
   - **Hidden Layers**: Gradual reduction through layers (e.g., 256 dimensions) with GELU activations and LayerNorm for stability
   - **Bottleneck Layer**: 128 dimensions, representing the final refined embeddings for each document

The autoencoder’s architecture is tailored to capture distinctive patterns in the dataset while reducing dimensionality, making embeddings more suitable for semantic retrieval.

#### Training Process
The autoencoder was trained using a combined loss function: **Mean Squared Error (MSE)** to ensure reconstruction fidelity, and **Cosine Similarity** to preserve semantic relationships. The model was trained for 50 epochs with a learning rate of 0.0001, weight decay of 1e-5, and a batch size of 16. A cosine annealing scheduler and gradient clipping were applied to improve learning stability and prevent overfitting.

---

### Step 2: Visualize Embeddings

To better understand the structure of our document embeddings, we projected them into a 2D space using t-SNE, both for pre-trained BERT embeddings and the embeddings adjusted by the autoencoder.

#### Visual Analysis
- **Clusters**: Clusters in the t-SNE visualization indicate potential topics or themes in the dataset.
- **Comparison**: The pre-trained and trained embeddings were compared to evaluate any increase in clustering quality, which could indicate better alignment with the document topics.

#### Clustering Metrics
The following metrics were used to evaluate clustering quality:
1. **Silhouette Score**: Measures how similar a point is to its own cluster versus other clusters (higher is better).
2. **Calinski-Harabasz Index**: Ratio of the sum of cluster dispersion to inter-cluster separation (higher is better).
3. **Davies-Bouldin Index**: Measures the average similarity ratio of each cluster with its most similar cluster (lower is better).

Results showed the following:

| Embeddings         | Silhouette Score | Calinski-Harabasz Index | Davies-Bouldin Index  |
|--------------------|------------------|-------------------------|-----------------------|
| Pre-trained BERT   | 0.3596           | 221.78                  | 0.86                  |
| Trained Embeddings | 0.4675           | 909.89                  | 0.71                  |

The autoencoder-adjusted embeddings slightly improved the Davies-Bouldin Index, indicating better-defined clusters.

---

### Step 3: Test the Search System

The final step was to build a search system using the embeddings and test it with different queries.

#### Search System
1. **Cosine Similarity** was used to compare the query embedding with document embeddings.
2. **Top N Results**: The system retrieves the top 10 most relevant documents based on cosine similarity.

#### Tests
Three different types of queries were used to evaluate the system:
1. **Test yielding 10 results**: Query `"users"`
2. **Test yielding less than 10 results**: Query `"love"`
3. **Test yielding non-obvious results**: Query `"writing"`

The system was able to retrieve relevant documents with appropriate results for each query type.

---

## Results

The trained embeddings provided more meaningful clustering and improved search relevance for this dataset. Using an autoencoder to adjust the BERT embeddings enabled more tailored document representations, which aligns with the project’s objectives.

---

## Usage

### Requirements
To run this project, you need the following libraries:
- `torch`
- `transformers`
- `sentence-transformers`
- `scikit-learn`
- `matplotlib`

You can install them with:
```bash
pip install torch transformers sentence-transformers scikit-learn matplotlib
```

### Running the Code

1. **Clone the repository**:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. **Check for Document Files**:
   Ensure that the `essays/` directory contains the required `.txt` documents. If missing, run:
```bash
python pg.py
```
   This script will download and save the necessary documents in the `essays/` directory.

3. **Generate Embeddings**:
   Train the embeddings with a pre-trained BERT model and Autoencoder:
```bash
python train_embeddings.py
```

4. **Visualize Embeddings**:
   Use t-SNE to visualize the pre-trained and adjusted embeddings:
```bash
python visualize_embeddings.py
```
   This will generate and save the visualizations in the `step2_output/` directory.

5. **Run Search Tests**:
   Test the query search system and view results for predefined queries:
```bash
python main.py
```
   This will run three search tests with queries (`users`, `love`, `writing`) and display the top 10 results for each.


