import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import random

# Model and training hyperparameters
class Config:
    MAX_LEN = 64
    BATCH_SIZE = 4
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    HIDDEN_DIM = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Initializing System on: {Config.DEVICE}...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Hybrid BERT-BiLSTM model with Attention
class HybridBertBiLSTMAttention(nn.Module):
    def __init__(self, n_classes):
        super(HybridBertBiLSTMAttention, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Bidirectional LSTM to capture sequential context
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=Config.HIDDEN_DIM,
                            batch_first=True,
                            bidirectional=True)

        # Linear layer to calculate attention scores
        self.attention_linear = nn.Linear(Config.HIDDEN_DIM * 2, 1)

        # Classification head
        self.classifier = nn.Linear(Config.HIDDEN_DIM * 2, n_classes)
        self.dropout = nn.Dropout(0.3)

    # Compute weighted sum of LSTM hidden states based on importance
    def attention_net(self, lstm_output):
        attn_weights = self.attention_linear(lstm_output).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(2)
        context_vector = torch.bmm(lstm_output.transpose(1, 2), attn_weights).squeeze(2)
        return context_vector, attn_weights

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        
        lstm_output, _ = self.lstm(sequence_output)
        context_vector, attn_weights = self.attention_net(lstm_output)

        output = self.classifier(self.dropout(context_vector))
        return output, attn_weights

# Sample dataset for classification
train_texts = [
    "You are stupid and ugly", "I will kill you", "You are a loser", "Go die", "You are an idiot",
    "dumb and annoying", "worst person ever", "shut up", "nobody likes you", "ugly face",
    "Have a nice day", "The weather is great", "I love this song", "You are amazing", "Good morning",
    "This is a test", "Hello friend", "What are you doing", "Keep it up", "Great job"
]
train_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# PyTorch Dataset wrapper
class DemoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            self.texts[item], max_length=Config.MAX_LEN, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.labels[item], dtype=torch.long)
        }

# Model initialization
print("Building Model...")
model = HybridBertBiLSTMAttention(n_classes=2).to(Config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Training loop
print("Training on Demo Data...")
dataset = DemoDataset(train_texts, train_labels, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
model.train()

for epoch in range(Config.EPOCHS):
    for batch in loader:
        input_ids = batch['input_ids'].to(Config.DEVICE)
        mask = batch['attention_mask'].to(Config.DEVICE)
        targets = batch['targets'].to(Config.DEVICE)

        optimizer.zero_grad()
        outputs, _ = model(input_ids, mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

print("Training Complete!")

# Inference function with attention visualization
def predict_sentiment(text):
    model.eval()
    encoded = tokenizer.encode_plus(text, max_length=Config.MAX_LEN, return_tensors='pt', padding='max_length', truncation=True)
    input_ids = encoded['input_ids'].to(Config.DEVICE)
    mask = encoded['attention_mask'].to(Config.DEVICE)

    with torch.no_grad():
        output, attn_weights = model(input_ids, mask)
        prediction = torch.argmax(output, dim=1).item()
        probs = F.softmax(output, dim=1)

        # Map attention weights back to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        weights = attn_weights[0].squeeze().cpu().numpy()

        # Filter special tokens for clean display
        word_weights = []
        for t, w in zip(tokens, weights):
            if t not in ['[PAD]', '[CLS]', '[SEP]']:
                word_weights.append((t, w))

        word_weights.sort(key=lambda x: x[1], reverse=True)

    return prediction, probs[0][prediction].item(), word_weights

# Main execution interface
print("\n" + "="*40)
print("CYBERBULLYING DETECTION SYSTEM READY")
print("Type 'exit' to quit.")
print("="*40 + "\n")

while True:
    user_input = input("Enter a sentence: ")
    if user_input.lower() == 'exit':
        break

    pred_class, confidence, top_words = predict_sentiment(user_input)

    label = "⚠️ ABUSIVE / BULLYING" if pred_class == 1 else "✅ NORMAL"
    color = "\033[91m" if pred_class == 1 else "\033[92m" 
    reset = "\033[0m"

    print(f"\nPrediction: {color}{label}{reset}")
    print(f"Confidence: {confidence:.2%}")
    print("Top Trigger Words (Attention):")
    for word, score in top_words[:3]: 
        print(f"  - {word}: {score:.4f}")
    print("-" * 30)
