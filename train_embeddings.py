import os
import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

# Carregar os textos dos artigos
essay_dir = "essays/"
documents = [open(os.path.join(essay_dir, f), 'r', encoding='utf-8').read()
             for f in os.listdir(essay_dir) if f.endswith(".txt")]

# Tokenizer e modelo BERT
print("🔄 Loading BERT model and tokenizer...\n")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
print("✔️ BERT model and tokenizer loaded successfully.\n")

# Função para criar embeddings dos documentos
def create_embeddings(documents):
    print("🔍 Generating embeddings for documents...\n")
    embeddings = []
    for idx, doc in enumerate(documents, 1):
        inputs = tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            doc_embedding = model(**inputs).last_hidden_state.mean(dim=1)
            embeddings.append(doc_embedding)
        if idx % 40 == 0 or idx == len(documents):
            print(f"   ➡️ Processed {idx}/{len(documents)} documents")
    print("✔️ Embeddings generated successfully.\n")
    return torch.cat(embeddings, dim=0)

# Definição do Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

if __name__ == "__main__":
    # Geração dos embeddings iniciais
    document_embeddings = create_embeddings(documents)

    input_dim = document_embeddings.shape[1]
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Configuração do DataLoader
    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings):
            self.embeddings = embeddings

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx]

    dataset = EmbeddingDataset(document_embeddings)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Treinamento do Autoencoder
    epochs = 10
    print("🚀 Starting training...\n")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"📈 Epoch {epoch + 1}/{epochs} completed - Average Loss: {avg_loss:.4f}")

    print("✔️ Training completed successfully.\n")

    # Passar os embeddings iniciais pelo autoencoder para obter os embeddings ajustados
    print("🔄 Generating adjusted embeddings...\n")
    with torch.no_grad():
        _, trained_embeddings = autoencoder(document_embeddings)
    print("✔️ Adjusted embeddings generated successfully.\n")

    # Salvar os embeddings ajustados e os embeddings originais
    output_dir = "embeddings/"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(autoencoder.state_dict(), os.path.join(output_dir, 'autoencoder_state.pth')) 
    torch.save(trained_embeddings, os.path.join(output_dir, 'trained_embeddings.pth'))  
    torch.save(document_embeddings, os.path.join(output_dir, 'document_embeddings.pth'))
    print("💾 Autoencoder state and embeddings saved successfully in 'embeddings' directory.\n")

