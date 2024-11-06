# Vector-Based Search System Using BERT and Autoencoder

This project implements a vector-based search system using BERT embeddings and an autoencoder to improve semantic relevance in document search. The goal is to create embeddings for each document and query, then use cosine similarity to retrieve the most relevant documents. This project was developed in multiple steps, as detailed below.

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

The main goal of this project is to represent both documents and queries in an embedding space using BERT, fine-tuned via an autoencoder, to allow for more semantically relevant search results. 

---

## Project Overview

In this system:
1. **BERT** is used to create embeddings for each document in the dataset.
2. **Autoencoder** is applied to further adjust the embeddings, compressing them into a lower-dimensional space to capture the dataset’s specific semantics.
3. **Cosine Similarity** is used to calculate the similarity between the query and document embeddings, retrieving the most relevant documents.

---

## Steps

### Step 1: Find Embeddings

In this initial step, we generated embeddings for each document in the dataset. 

1. **Dataset**: The dataset consists of a collection of blog posts by Paul Graham.
2. **BERT Model**: We used the `bert-base-nli-mean-tokens` pre-trained model from the `sentence-transformers` library to generate the initial embeddings.
3. **Autoencoder**: An autoencoder with the following structure was applied to compress the embeddings:
   - Input Layer: 768 dimensions (BERT embedding size)
   - Hidden Layer: 256 dimensions
   - Bottleneck (Embedding) Layer: 128 dimensions

The autoencoder’s purpose is to create an embedding space more tailored to this dataset, enhancing retrieval quality by capturing specific features.

#### Training Process
The autoencoder was trained using Mean Squared Error (MSE) as the loss function, with 10 epochs and a learning rate of 0.001.

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

| Embeddings         | Silhouette Score | Calinski-Harabasz Index | Davies-Bouldin Index |
|--------------------|------------------|-------------------------|-----------------------|
| Pre-trained BERT   | 0.0933          | 11.33                   | 3.29                  |
| Trained Embeddings | 0.0877          | 10.13                   | 2.98                  |

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

### Running the Code

1. **Generate Embeddings**:
   ```bash
   python train_embeddings.py
