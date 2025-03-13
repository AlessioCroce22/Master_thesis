import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

'''
Parte di caricamento dei loader per train, validation e test
'''

class ECGDataset(Dataset):
    def __init__(self, data_folder, csv_file, pad_value=-999, step=100):
        self.data_folder = data_folder
        self.data_info = pd.read_csv(csv_file)  # NON mescoliamo il CSV
        self.pad_value = pad_value  # Valore di padding per la maschera
        self.step = step  # Passo di sottocampionamento

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        file_name = self.data_info.iloc[idx, 0]  # Nome file .npy
        label = self.data_info.iloc[idx, 1]  # Etichetta
        file_path = os.path.join(self.data_folder, file_name)

        # Carichiamo il segnale ECG
        data = np.load(file_path)  # Shape: (lunghezza, canali)
        data = data[::self.step]  # Sottocampionamento prendendo un valore ogni 'step'
        data = torch.tensor(data, dtype=torch.float32)

        # Controlliamo se ci sono NaN nei dati
        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"❌❌❌ NaN o inf nei dati di input! File: {file_name}")
            print(f"Min: {data.min().item()}, Max: {data.max().item()}")
            return None  # Salta il batch

        # Calcoliamo l'Attention Mask: 1 per valori validi, 0 per padding
        attention_mask = (data != self.pad_value).all(dim=1).int()


        label = torch.tensor(label - 1, dtype=torch.long)

        return data, attention_mask, label

# Creiamo il dataset SENZA mescolare i dati
full_dataset = ECGDataset(r"/app/work-data/Only-ECG-raw/train_log", r"/app/work-data/Only-ECG-raw/train_log/train_labels.csv", pad_value=-999)
test_dataset= ECGDataset(r"/app/work-data/Only-ECG-raw/test_log", r"/app/work-data/Only-ECG-raw/test_log/test_labels.csv", pad_value=-999)

# **📌 Split Deterministico (80% Train - 20% Validation)**
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)  # Primi 80% per il Train
val_size = dataset_size - train_size  # Ultimi 20% per la Validation

train_dataset = Subset(full_dataset, list(range(train_size)))  # Indici 0 → train_size-1
val_dataset = Subset(full_dataset, list(range(train_size, dataset_size)))  # Indici train_size → fine

# Creiamo DataLoader per Train, Validation e Test
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)  # No shuffle!
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)  # No shuffle!
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=0)


'''
Modello del trasformer
'''

# Positional Encoding (aggiunge info temporali ai segnali)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=11960, pad_value=-999):
        super(PositionalEncoding, self).__init__()
        self.pad_value = pad_value
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)

    def forward(self, x):
        # Creiamo una maschera per i token di padding
        mask = x != self.pad_value
        x = x.masked_fill(~mask, 0)  # Imposta a zero i padding per evitare contaminazione
        
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # Aggiunge la posizione solo ai valori validi
        x = x * mask  # Rimuove le modifiche ai token di padding

        return x

# Transformer Encoder con Pre-LN
class TransformerEncoderLayerPreLN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.5, pad_value=-999):
        super(TransformerEncoderLayerPreLN, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.pad_value = pad_value

    def forward(self, src, src_mask):
        mask = src != self.pad_value  # Mask dei valori validi

        #print(f"\n🔹 PRIMA DI LAYER NORM 1: Min {src.min().item()}, Max {src.max().item()}")
        src_norm = self.norm1(src.masked_fill(~mask, 0))  # Ignora i padding nella normalizzazione
        #print(f"🔹 DOPO LAYER NORM 1: Min {src_norm.min().item()}, Max {src_norm.max().item()}")
        #print(f"🔍 UNIQUE VALUES IN src_mask: {torch.unique(src_mask)}")

        #src_mask = src_mask.to(dtype=torch.bool)
        src_mask = ~src_mask.to(dtype=torch.bool)
        #print(f"🔍 UNIQUE VALUES IN src_mask: {torch.unique(src_mask)}")
        #print(f"📌 PRIMA DI ATTENTION: Min {src_norm.min().item()}, Max {src_norm.max().item()}, Has NaN: {torch.isnan(src_norm).any().item()}")


        # Decomposizione della MultiheadAttention
        Q = src_norm.clone()
        K = src_norm.clone()
        V = src_norm.clone()

        # Matrice di similarità (QK^T)
        QK_T = torch.matmul(Q, K.transpose(-2, -1))
        #print(f"🎭 QK^T Min: {QK_T.min().item()}, Max: {QK_T.max().item()}, Has NaN: {torch.isnan(QK_T).any().item()}")

        # Controllo delle distribuzioni di Q, K, V
        #print(f"🟢 Q Min: {Q.min().item()}, Max: {Q.max().item()}, Has NaN: {torch.isnan(Q).any().item()}")
        #print(f"🔵 K Min: {K.min().item()}, Max: {K.max().item()}, Has NaN: {torch.isnan(K).any().item()}")
        #print(f"🟣 V Min: {V.min().item()}, Max: {V.max().item()}, Has NaN: {torch.isnan(V).any().item()}")



        attn_output, attn_weights = self.self_attn(src_norm, src_norm, src_norm, key_padding_mask=src_mask)
        #print(f"🔹 DOPO ATTENTION: Min {attn_output.min().item()}, Max {attn_output.max().item()}")
        #print(f"🎯 Attention Weights Min: {attn_weights.min().item()}, Max: {attn_weights.max().item()}, Has NaN: {torch.isnan(attn_weights).any().item()}")

        src = src + self.dropout(attn_output)
        #print(f"🔹 DOPO ADD & DROPOUT 1: Min {src.min().item()}, Max {src.max().item()}")

        src_norm = self.norm2(src.masked_fill(~mask, 0))
        #print(f"🔹 DOPO LAYER NORM 2: Min {src_norm.min().item()}, Max {src_norm.max().item()}")

        ff_output = self.feedforward(src_norm)
        #print(f"🔹 DOPO FEEDFORWARD: Min {ff_output.min().item()}, Max {ff_output.max().item()}")

        src = src + self.dropout(ff_output)
        #print(f"🔹 DOPO ADD & DROPOUT 2: Min {src.min().item()}, Max {src.max().item()}")

        return src * mask  # Riapplica la maschera


# Modello Transformer per ECG
class ECGTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=128, num_layers=1, nhead=4, num_classes=9, pad_value=-999):
        super(ECGTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerPreLN(d_model, nhead, pad_value=pad_value) for _ in range(num_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
        self.pad_value = pad_value

    def forward(self, x, mask):
        x = self.input_proj(x)

        # Rende il padding neutro prima di aggiungere il Positional Encoding
        x = x.masked_fill(x == self.pad_value, 0)
        x = self.pos_encoder(x)  # Il nuovo Positional Encoding già gestisce i padding

        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, mask)

        x = x.permute(1, 2, 0)  # (batch, d_model, seq_len)

        # 🔹 Adattiamo mask alla shape corretta
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)

        mask_sum = mask.sum(dim=2, keepdim=True).clamp(min=1)  # Ora ha shape (batch, 1, 1)

        # ⬇️ Aggiungi questa riga per azzerare i token di padding prima della media
        x = x * mask  # Evita che i padding contaminino la media

        x = x.sum(dim=-1, keepdim=True) / mask_sum  # Ora la divisione ha shape corretta

        x = x.squeeze(-1)  # Rimuove l'ultima dimensione in eccesso

        x = self.fc(x)

        return x






'''
Train part
'''       

device = torch.device("cuda:0")

# Stampa il nome della GPU
gpu_name = torch.cuda.get_device_name(device)
print(f"Stai usando la GPU: {gpu_name}")

# Inizializza il modello
model = ECGTransformer(num_classes=9).to(device)

# Loss, Ottimizzatore e Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scaler = torch.amp.GradScaler('cuda')  # Per precisione mista

def train_model(model, train_loader, val_loader, epochs=100, patience=20):
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train, total_train = 0, 0

        for batch_idx, (inputs, masks, labels) in enumerate(train_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):  # Precisione mista
                outputs = model(inputs, masks)
                loss = criterion(outputs, labels)

            if torch.isnan(loss).any():
                print("\n❌❌❌ NaN DETECTED IN LOSS! ❌❌❌")
                return

            scaler.scale(loss).backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ⬅️ Gradient Clipping
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct_train / total_train

        # 📌 Validation
        model.eval()
        val_loss = 0
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for inputs, masks, labels in val_loader:
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                outputs = model(inputs, masks)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val

        scheduler.step()
        for param_group in optimizer.param_groups:
            print(f"📉 Learning Rate: {param_group['lr']}")

        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 📌 Early Stopping & Model Saving
        if val_loss < best_val_loss or val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

train_model(model, train_loader, val_loader)


'''
test part
'''


# Carica il miglior modello salvato
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Funzione per il Test
def test_model(model, test_loader):
    correct_test, total_test = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, masks, labels in test_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            outputs = model(inputs, masks)

            preds = outputs.argmax(dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct_test / total_test
    print(f"Test Accuracy: {test_acc:.4f}")

    return all_preds, all_labels

# 📌 Esegui il test
test_preds, test_labels = test_model(model, test_loader)