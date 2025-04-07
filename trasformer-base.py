import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch.utils.checkpoint as cp
import time
from transformers import LongformerModel, LongformerConfig
import copy
from tqdm import tqdm
from torchvision.models import resnet18
torch.backends.cudnn.benchmark = True
class SignalDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        signal = data[:-1]  # Tutto tranne l'ultimo valore
        label = data[-1]  # Ultimo valore
        
        mask = (signal != -100).astype(np.float32)  # 1 per valori validi, 0 per padding
        signal = signal * mask  # Mantiene solo i valori validi
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def create_dataloaders(training_path, test_path, batch_size=32):
    train_files = [os.path.join(training_path, f) for f in os.listdir(training_path) if f.endswith('.npy')]
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.npy')]
    
    total_size = len(train_files) + len(test_files)
    train_size = int(0.75 * total_size)
    val_size = int(0.05 * total_size)
    
    train_files, val_files = train_test_split(train_files, test_size=val_size, stratify=[np.load(f)[-1] for f in train_files], random_state=42)
    
    train_dataset = SignalDataset(train_files)
    val_dataset = SignalDataset(val_files)
    test_dataset = SignalDataset(test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=False)
    
    return train_loader, val_loader, test_loader

def verify_dataloaders(train_loader, val_loader):
    print("Verifying dataset integrity...")
    train_labels = []
    val_labels = []
    test_labels = []
    for signals, masks, labels in train_loader:
        assert all(len(signal) == 15000 for signal in signals), "Error: Signal length mismatch! in train"
        assert torch.all(signals * (1 - masks) == 0), "Error: Mask does not correctly zero out padded values! in train"
        train_labels.extend(labels.numpy())
    
    for signals, masks, labels in val_loader:
        assert all(len(signal) == 15000 for signal in signals), "Error: Signal length mismatch! in val"
        assert torch.all(signals * (1 - masks) == 0), "Error: Mask does not correctly zero out padded values! in val"
        val_labels.extend(labels.numpy())

    for signals, masks, labels in test_loader:
        assert all(len(signal) == 15000 for signal in signals), "Error: Signal length mismatch! in test"
        assert torch.all(signals * (1 - masks) == 0), "Error: Mask does not correctly zero out padded values! in test"
        test_labels.extend(labels.numpy())
    
    train_label_counts = Counter(train_labels)
    val_label_counts = Counter(val_labels)
    test_label_counts = Counter(test_labels)

    print("Training label distribution:", train_label_counts)
    print("Validation label distribution:", val_label_counts)
    print("Test label distribution:", test_label_counts)

    train_total = sum(train_label_counts.values())
    val_total = sum(val_label_counts.values())
    test_total = sum(test_label_counts.values())

    for label in train_label_counts:
        train_ratio = train_label_counts[label] / train_total
        val_ratio = val_label_counts.get(label, 0) / val_total
        test_ratio = test_label_counts.get(label,0) /test_total
        print(f"Label {label}: Train Ratio = {train_ratio:.4f}, Validation Ratio = {val_ratio:.4f}, Test Ratio = {test_ratio:.4f}")


#MIX dei due modelli base e potente

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_embed = self.pe(positions)  # (1, T, D)
        return x + pos_embed

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_weights = nn.Linear(d_model, 1)
    
    def forward(self, x, mask=None):
        attn_scores = self.attn_weights(x).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        pooled_output = torch.sum(x * attn_weights, dim=1)
        return pooled_output
class LinearAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / Q.size(-1)**0.5  # (B, T, T)

        if mask is not None:
            attn_mask = mask.unsqueeze(1) == 0  # (B, 1, T)
            attn_scores = attn_scores.masked_fill(attn_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)
        attn_output = torch.bmm(attn_weights, V)  # (B, T, D)

        return self.out_proj(attn_output)

def masked_mean(x, mask):
    mask = mask.unsqueeze(-1)  # (B, T, 1)
    summed = (x * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)
    return summed / count

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.self_attn = LinearAttention(d_model)  # sostituito

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):

        attn_output = self.self_attn(src, mask=~src_mask) if src_mask is not None else self.self_attn(src)

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src



class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x  # leggero skip
        x = self.conv3(x)
        return x + residual  # light residual

class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, num_heads=4, num_layers=2, num_classes=4, dropout=0.3, pooling_type="attention"):
        super().__init__()
        #self.feature_extractor = nn.Sequential(
        #    nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
        #    nn.ReLU(),
        #    nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2)
        #)
        self.feature_extractor = CNNFeatureExtractor(d_model)

        #self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        #self.pos_encoding = LearnedPositionalEncoding(max_len=15000, d_model=d_model)

        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, num_heads, 4 * d_model, dropout)
            for _ in range(num_layers)
        ])

        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.pooling = AttentionPooling(d_model)
        elif pooling_type == "mean":
            self.pooling = None  # handled manually in forward
        else:
            raise ValueError("pooling_type must be 'attention' or 'mean'")

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask):
        batch_size, seq_len = x.shape
        #x = x.unsqueeze(-1)
        #x = self.proj(x)
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.feature_extractor(x)  # (B, d_model, T)
        x = x.permute(0, 2, 1)  # (B, T, d_model)

        x = self.pos_encoding(x)

        attn_mask = (mask == 0).to(torch.bool)
        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)

        
        if self.pooling_type == "attention":
            x = self.pooling(x, mask)
        else:
            x = masked_mean(x, mask)

        logits=self.fc(x)
        
        return logits

# Se ci sono più GPU, usa DataParallel
def setup_model(model):
    model.cuda()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    return model

# Funzione di training
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, patience=5):
    model = setup_model(model)

    # Ottimizzatore e Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
    #scaler = torch.amp.GradScaler(device="cuda")

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_val=200.0
    patience_counter = 0
    
    #param_count=0
    #for name, param in model.named_parameters():
    #    print(name)
    #    param_count+=1
    #print(f"learnable parameters: {param_count}")

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n🟢 Epoch {epoch+1}/{num_epochs} - Training...")
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, ncols=100)

        # ---------------- TRAINING --------------
        train_loss = 0.0
        all_preds, all_labels = [], []
        cont=0
        for signals, masks, labels in train_loader_tqdm:
            #if cont % 50 ==0:
            #print(f"batch numero: {cont+1}")
            signals, masks, labels = signals.cuda(device, non_blocking=True), masks.cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)
            
            optimizer.zero_grad()
            #with torch.amp.autocast(device_type="cuda"):
            logits = model(signals, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            #scaler.scale(loss).backward()  # Scala la loss
            #scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            #scaler.step(optimizer)  # Aggiorna i pesi
            #scaler.update()  # Aggiorna il GradScaler
            for name, param in model.named_parameters():
                if param.grad is not None:
                    max_grad = param.grad.abs().max()
                    if torch.isnan(max_grad) or torch.isinf(max_grad):
                        print(f"loss: {loss}")
                        print(f"logits: {logits}")
                        print(f"labels: {labels}")
                        print(f"preds: {torch.argmax(logits, dim=1)}")
                        print(f"❌ NaN/Inf nei gradienti di {name}")
                    elif max_grad > 1e3:
                        print(f"⚠️ Gradiente molto alto ({max_grad.item():.2e}) in {name}")
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            cont+=1
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        avg_train_loss = train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        print(f"validation epoch {epoch+1}")
        with torch.no_grad():
            for signals, masks, labels in val_loader:
                signals, masks, labels = signals.cuda(device, non_blocking=True), masks.cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)
                
                logits = model(signals, masks)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        class_f1 = f1_score(all_labels, all_preds, average=None)  # F1 per classe
        avg_val_loss = val_loss / len(val_loader)
        conf_matrix = confusion_matrix(all_labels, all_preds)



        # Scheduler step
        scheduler.step()

        # Early Stopping check
        if val_f1> best_f1 :
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Stampa risultati
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        print(f"Class-wise F1: {class_f1}")
        print(f"Confusion Matrix: {conf_matrix}")
    # Carica il modello migliore
    model.load_state_dict(best_model_wts)
    return model

# Funzione di test
def test_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, masks, labels in test_loader:
            signals, masks, labels = signals.cuda(device, non_blocking=True), masks.cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)

            logits = model(signals, masks)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_f1 = f1_score(all_labels, all_preds, average="macro")
    class_f1 = f1_score(all_labels, all_preds, average=None)

    print("\nTest Results:")
    print(f"Overall F1-score: {test_f1:.4f}")
    print(f"Class-wise F1: {class_f1}")


print("inizio")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Esempio di utilizzo
training_path = "/app/work-data/training"
test_path = "/app/work-data/test"
batch_size = 2
train_loader, val_loader, test_loader = create_dataloaders(training_path, test_path, batch_size)
#verify_dataloaders(train_loader, val_loader)


# Controllo GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Inizializza il modello
model = ECGTransformer(pooling_type="attention")

# Addestramento
best_model = train_model(model, train_loader, val_loader, num_epochs=300, lr=5e-4, patience=100)
print("test")
# Test finale
test_model(best_model, test_loader)
#tutte prove con un terzo di segnale dropout a 0.2, weight decay 1e-4 lr 1e-3
#MODELLO attuale ma con proj al posto di CNN fa schifo tipo 30%

#PRIMA #prova con mean pooling, cnn due layer 16 dim, dmodel 64, n heads 2 layer 2 fa 48%

#SECONDA PROVA con attention pooling stesse stats fa 45%

#Prima prova con attention pooling, cnn potenziata, il resto uguale 50%

#seconda prova con mean, cnn potenziata e positional enc diverso, il resto uguale 45%

#prima prova con attention, cnn potenziata e enc diverso 48%

#seconda prova con cnn potenziata e mean 49%

#prima di queste vedere i learnable parameters di modello migliore e cnn: 
#CNN 62 param, trasf nella configurazione piu pesante: CNN, learnable pos, attention pool 51

#prima prova con label smoothing, dropuot a 0.4 e config piu pesante fa 49%

#seconda prova label smoothing, dropout 0.4 e conifg attention pool piu cnn forte fa 49%

#prima prova con label smoothing, dropuot a 0.4, lr a 5e-4 e weight decay a 1e-3,
# nhead 4 dmodel 128 layer 2, cnn potente e positional encoding, attention pooling : 51% ma con piu epoche forse faceva meglio

#seconda prova con label smoothing, dropuot a 0.4, lr a 5e-4 e weight decay a 1e-3,
# nhead 4 dmodel 128 layer 2, cnn scarsa e positional encoding, attention pooling : 42% ma con piu epoche faceva meglio

#prima prova uguale alla seconda di prima quindi cnn potente ma con dropout a 0.3 e weight decay a 5e-4 e lr a 1e-3 + customtrasformer nuovo: fa cagare 41% 

#stanotte fare 3 prove in parallelo con segnale intero: prima prova cnn pompata, pos encoding semplice, dropout 0.2, weight decay 1e-4 lr 5e-4 e trasformer leggero quindi 64,2,2
#seconda prova cnn potente pos encoding trasformer via di mezzo con 128,2,2 regolarizzazioni drop 0.3 weight 1e-4 lr 5e-4
#terza prova cnn potente, pos encoding, trasformer pompato regolarizzazioni drop 0.3 weight 1e-4 lr 5e-4

#da provare le stesse 3 config ma con la cnn scarsa se overfitta pesante