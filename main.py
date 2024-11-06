import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
from train_embeddings import Autoencoder

print("🔄 Loading BERT model and tokenizer...\n")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
print("✔️ BERT model and tokenizer loaded successfully.\n")

# Carregar o Autoencoder e seu estado treinado
autoencoder = Autoencoder(input_dim=768)  # Dimensão original do BERT
autoencoder.load_state_dict(torch.load('embeddings/autoencoder_state.pth'))
autoencoder.eval()

# Carregar os embeddings ajustados e os títulos dos documentos
trained_embeddings = torch.load('embeddings/trained_embeddings.pth')
titles = [title.replace('.txt', '') for title in os.listdir('essays/') if title.endswith('.txt')]
print(f"📄 Loaded {len(titles)} documents from 'essays/' directory.\n")

# Função para gerar o embedding da consulta e ajustá-lo com o autoencoder
def generate_query_embedding(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1)
        _, query_embedding = autoencoder(query_embedding)  # Ajuste para 128 dimensões
    return query_embedding

# Função para buscar documentos relevantes
def search_documents(query, top_n=10):
    print(f"🔍 Searching for query: '{query}'")
    query_embedding = generate_query_embedding(query).numpy()
    
    # Calcular a similaridade cosseno entre a consulta e os embeddings ajustados
    similarities = cosine_similarity(query_embedding, trained_embeddings.numpy())[0]

    # Ordenar os documentos pela relevância e selecionar os top_n resultados
    sorted_indices = similarities.argsort()[::-1][:top_n]
    results = [{"title": titles[idx], "similarity": round(float(similarities[idx]), 4)} for idx in sorted_indices]

    print(f"\n📝 Top {top_n} results for '{query}':")
    print("-" * 50)
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} - Similarity: {result['similarity']}")
    print("-" * 50 + "\n")
    return results

# Testes de busca com formatação aprimorada
print("🔎 Running search tests...\n")

print("💡 Test with 10 results for query 'users':")
search_documents("users", top_n=10)

print("💡 Test with less than 10 results for query 'love':")
search_documents("love", top_n=10)

print("💡 Test with a non-obvious result for query 'writing':")
search_documents("writing", top_n=10)
