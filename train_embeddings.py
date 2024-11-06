import os
import torch
import re
from torch import nn, optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Função de pré-processamento de texto
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Carregar documentos e pré-processar
essay_dir = "essays/"
documents = [open(os.path.join(essay_dir, f), 'r', encoding='utf-8').read()
             for f in os.listdir(essay_dir) if f.endswith(".txt")]

print("📄 Preprocessing text documents...\n")
documents = [preprocess_text(doc) for doc in documents]
print("✔️ Text documents preprocessed successfully.\n")

# Tokenizer e modelo BERT
print("🔄 Loading BERT model and tokenizer...\n")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
print("✔️ BERT model and tokenizer loaded successfully.\n")

# Função para criar embeddings dos documentos com máscara de atenção
def create_embeddings(documents):
    print("🔍 Generating embeddings for documents...\n")
    embeddings = []
    for idx, doc in enumerate(documents, 1):
        inputs = tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1) 
            doc_embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(doc_embedding)
        if idx % 40 == 0 or idx == len(documents):
            print(f"   ➡️ Processed {idx}/{len(documents)} documents")
    print("✔️ Embeddings generated successfully.\n")
    return torch.cat(embeddings, dim=0)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

# Definição do Autoencoder com Dropout, LayerNorm e GELU
class SemanticAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(SemanticAutoencoder, self).__init__()
        
        h1_dim = input_dim // 2
        h2_dim = h1_dim // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1_dim),
            nn.LayerNorm(h1_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(h1_dim, h2_dim),
            nn.LayerNorm(h2_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(h2_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, h2_dim),
            nn.LayerNorm(h2_dim),
            nn.GELU(),
            
            nn.Linear(h2_dim, h1_dim),
            nn.LayerNorm(h1_dim),
            nn.GELU(),
            
            nn.Linear(h1_dim, input_dim)
        )

    def add_noise(self, x, noise_factor=0.05):
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    def forward(self, x, add_noise=True):
        if add_noise and self.training:
            x = self.add_noise(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Função de perda híbrida para preservação semântica
class SemanticLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(SemanticLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
        self.alpha = alpha

    def forward(self, decoded, encoded, original, autoencoder):
        mse_loss = self.mse(decoded, original)
        
        with torch.no_grad():
            original_encoded = autoencoder.encoder(original)
        
        target = torch.ones(encoded.size(0), device=encoded.device)
        
        semantic_loss = self.cosine_loss(encoded, original_encoded, target)
        
        return self.alpha * mse_loss + (1 - self.alpha) * semantic_loss

def train_model(autoencoder, train_loader, criterion, optimizer, scheduler, epochs=50):
    print("🚀 Starting training with semantic preservation...\n")
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            reconstructed, encoded = autoencoder(data)
            loss = criterion(reconstructed, encoded, data, autoencoder)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"📈 Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        # Early stopping com salvamento do melhor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    document_embeddings = create_embeddings(documents)
    input_dim = document_embeddings.shape[1]
    autoencoder = SemanticAutoencoder(input_dim)
    
    criterion = SemanticLoss(alpha=0.7)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    dataset = EmbeddingDataset(document_embeddings)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    train_model(autoencoder, train_loader, criterion, optimizer, scheduler)
    
    autoencoder.load_state_dict(torch.load('best_model.pth'))
    autoencoder.eval()
    
    with torch.no_grad():
        _, trained_embeddings = autoencoder(document_embeddings)
        trained_embeddings = F.normalize(trained_embeddings, p=2, dim=1)
    
    # Salvar embeddings
    output_dir = "embeddings/"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(autoencoder.state_dict(), os.path.join(output_dir, 'autoencoder_state.pth'))
    torch.save(trained_embeddings, os.path.join(output_dir, 'trained_embeddings.pth'))
    torch.save(document_embeddings, os.path.join(output_dir, 'document_embeddings.pth'))
    print("💾 Autoencoder state and embeddings saved successfully in 'embeddings' directory.\n")
