import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import gc
cudnn.benchmark = True

'''
Parte di caricamento dei loader per train, validation e test
'''

class ECGDataset(Dataset):
    def __init__(self, data_folder, csv_file, pad_value=-999, step=1, chunk_size=4000):
        self.data_folder = data_folder
        self.data_info = pd.read_csv(csv_file)
        self.pad_value = pad_value
        self.step = step
        self.chunk_size = chunk_size


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        file_name = self.data_info.iloc[idx, 0]
        label = self.data_info.iloc[idx, 1]
        file_path = os.path.join(self.data_folder, file_name)

        data = np.load(file_path)  # Shape: (lunghezza, canali)
        data = data[::self.step]  # Sottocampionamento se necessario

        # Suddividiamo il segnale in finestre (chunk)
        chunks = []
        masks = []
        num_chunks = math.ceil(len(data) / self.chunk_size)

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = data[start:end]

            # Se il chunk è più corto, facciamo il padding
            if len(chunk) < self.chunk_size:
                pad_length = self.chunk_size - len(chunk)
                padding = np.full((pad_length, chunk.shape[1]), self.pad_value)
                chunk = np.vstack((chunk, padding))

            chunk = torch.tensor(chunk, dtype=torch.float32)
            mask = (chunk != self.pad_value).all(dim=1).int()

            chunks.append(chunk)
            masks.append(mask)
        chunks=torch.stack(chunks)
        masks=torch.stack(masks)
        label = torch.tensor(label - 1, dtype=torch.long)

        return chunks, masks, label  # Restituisce tutti i chunk, maschere e label

    def print_label_distribution(self):
        labels = self.data_info.iloc[:, 1] - 1  # Convertiamo le etichette per iniziare da 0
        label_counts = Counter(labels)
        
        print("\nDistribuzione delle etichette nei campioni:")
        for label, count in sorted(label_counts.items()):
            print(f"Label {label}: {count} campioni")

class ChunkedECGDataset(Dataset):
    """
    Dataset che restituisce un singolo chunk alla volta per il training e la validation.
    Assicura che ogni chunk venga visto una sola volta in ordine randomico.
    """
    def __init__(self, base_dataset,train):
        self.base_dataset = base_dataset
        self.chunk_indices = []  # Lista di tuple (sample_idx, chunk_idx)
        self.train=train
        self.lables=[]
        for sample_idx in range(len(base_dataset)):
            chunks, masks, label = base_dataset[sample_idx]
            num_chunks = chunks.shape[0]  # Numero di chunk disponibili per questo sample

            for chunk_idx in range(num_chunks):
                chunk = chunks[chunk_idx]
                # Controlliamo se il chunk è composto solo da -999
                if not torch.all(chunk == -999):  # Se contiene almeno un valore diverso, lo teniamo
                    self.chunk_indices.append((sample_idx, chunk_idx))
                    self.lables.append(label)

    def __len__(self):
        return len(self.chunk_indices)  # Numero totale di chunk disponibili
    def getlabels(self):
        return self.lables
    def __getitem__(self, idx):
        sample_idx, chunk_idx = self.chunk_indices[idx]
        chunks, masks, label = self.base_dataset[sample_idx]

        chunk = chunks[chunk_idx]  # Prendiamo solo il chunk specificato
        mask = masks[chunk_idx]  # Prendiamo la sua maschera
        
            
        if self.train==True:
            # Creiamo una maschera booleana per i valori validi (senza padding)
            valid_indices = mask.bool()  # Shape: (seq_len,)

            if torch.rand(1).item() < 0.5:
                noise = torch.randn_like(chunk) * 0.02  # Jittering con rumore gaussiano
                chunk[valid_indices, :] += noise[valid_indices, :]

            if torch.rand(1).item() < 0.3:
                scale = torch.empty(1).uniform_(0.8, 1.2)  # Scaling casuale
                chunk[valid_indices, :] *= scale
            
        return chunk, mask, label  # Restituiamo solo un chunk alla volta


# Creiamo il dataset di base (carica tutti i segnali)
base_train_val_dataset = ECGDataset(
    r"/app/work-data/Only-ECG-raw/train_log",
    r"/app/work-data/Only-ECG-raw/train_log/train_labels.csv",
    pad_value=-999
)

dataset_size = len(base_train_val_dataset)
train_size = int(0.8 * dataset_size)  # Primi 80% per il Train
val_size = dataset_size - train_size  # Ultimi 20% per la Validation

train_dataset = Subset(base_train_val_dataset, list(range(train_size)))  # Indici 0 → train_size-1
val_dataset = Subset(base_train_val_dataset, list(range(train_size, dataset_size)))  # Indici train_size → fine

base_test_dataset = ECGDataset(
    r"/app/work-data/Only-ECG-raw/test_log",
    r"/app/work-data/Only-ECG-raw/test_log/test_labels.csv",
    pad_value=-999
)

# Espandiamo il dataset per training e validation (restituisce un chunk alla volta)
train_dataset = ChunkedECGDataset(train_dataset,True)
val_dataset = ChunkedECGDataset(val_dataset,False)

# Test set rimane invariato (restituisce interi segnali con tutti i chunk)
test_dataset = base_test_dataset

# 1️⃣ Estrazione delle etichette dal dataset
labels = np.array(train_dataset.getlabels())

# 2️⃣ Conta il numero di campioni per classe
class_counts = Counter(labels)
total_samples = sum(class_counts.values())

alpha1=0.6
alpha2=0.2
# 3️⃣ Calcola i pesi per ogni classe (peso = totale campioni / numero di campioni per classe)
class_weights = {cls: (total_samples / count) ** alpha1 for cls, count in class_counts.items()}
class_weights_w = {cls: (total_samples / count) ** alpha2 for cls, count in class_counts.items()}
# 5️⃣ Creazione dei pesi per ogni campione (usando np.vectorize per velocizzare)
sample_weights_tensor = torch.tensor(np.vectorize(class_weights.get)(labels),dtype=torch.float32)

# 6️⃣ Convertiamo in tensore PyTorch
weights_tensor = torch.tensor(
    [class_weights_w[cls] for cls in sorted(class_counts.keys())], dtype=torch.float32
)

# 7️⃣ Creiamo il WeightedRandomSampler
sampler = WeightedRandomSampler(
    weights=sample_weights_tensor,
    num_samples=len(sample_weights_tensor),
    replacement=True  # Permettiamo il campionamento con sostituzione
)

# 🔹 Controlliamo i valori
# Creiamo DataLoader
train_loader = DataLoader(train_dataset, batch_size=48, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)  # ⚠️ Nessun shuffle per il test

'''
Modello del trasformer
'''

# Positional Encoding (aggiunge info temporali ai segnali)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4000, pad_value=-999):
        super(PositionalEncoding, self).__init__()
        self.pad_value = pad_value
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)

    def forward(self, x,mask):
        # Creiamo una maschera per i token di padding
        x = x.masked_fill(~mask, 0)  # Imposta a zero i padding per evitare contaminazione
        
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # Aggiunge la posizione solo ai valori validi
        x = x * mask  # Rimuove le modifiche ai token di padding

        return x

# Transformer Encoder con Pre-LN
class TransformerEncoderLayerPreLN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.5, pad_value=0.0):
        super(TransformerEncoderLayerPreLN, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,batch_first=True,dropout=dropout)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(dim_feedforward, d_model)
        )
        self.nhead=nhead
        self.dropout = nn.Dropout(dropout)
        self.pad_value = 0.0

    def forward(self, src,value_mask,src_mask,id_batch):
        if torch.isnan(src).any():
            print("NaN in layer")
            return
        valid_counts = value_mask.sum(dim=-1, keepdim=True)

        # Evitiamo divisioni per zero sostituendo i casi invalidi con 1 (così non cambia il valore finale)
        valid_counts = valid_counts.clamp(min=1)  # Evita 0/0

        # Calcoliamo media e varianza escludendo i padding
        mean = (src * value_mask).sum(dim=-1, keepdim=True) / valid_counts
        var = ((src - mean) * value_mask).pow(2).sum(dim=-1, keepdim=True) / valid_counts
        if torch.isnan(mean).any():
            print("NaN in mean")
            return
        if torch.isnan(var).any():
            print("NaN in var")
            return    
        src_norm = (src - mean) / (var + 1e-5).sqrt()
        if torch.isnan(src_norm).any():
            print("NaN in src_norm")
            return 
        # Rimettiamo a zero i padding
        src_norm = src_norm * value_mask
        src_mask = ~src_mask
        
        attn_mask = src_mask.unsqueeze(1) * src_mask.unsqueeze(2)  # Shape (batch, seq_len, seq_len)
        attn_mask = attn_mask.float()
        attn_mask = attn_mask.masked_fill(attn_mask == 1, -1e9)  # Blocca interazioni con padding
        attn_mask = attn_mask.masked_fill(attn_mask != -1e9, 0)  # Assicura che gli altri valori siano 0
        attn_mask = attn_mask.repeat(self.nhead, 1, 1)
        
        attn_output, attn_weights = self.self_attn(src_norm, src_norm, src_norm, attn_mask=attn_mask)
        
        attn_output=attn_output.nan_to_num(0)
        attn_weights=attn_weights.nan_to_num(0)   
        attn_output=attn_output*value_mask    
        
        src = src + self.dropout(attn_output)
        
        if torch.isnan(src).any():
            print("NaN in src dopo attention")

        mean = (src * value_mask).sum(dim=-1, keepdim=True) / valid_counts
        var = ((src - mean) * value_mask).pow(2).sum(dim=-1, keepdim=True) / valid_counts

        src_norm = (src - mean) / (var + 1e-5).sqrt()
        
        # Rimettiamo a zero i padding
        src_norm = (src_norm * value_mask)    
        
        ff_output = self.feedforward(src_norm)
        
        
        src = src + self.dropout(ff_output)
        
            
        return src  # Riapplica la maschera


# Modello Transformer per ECG
class ECGTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=48, num_layers=1, nhead=1, num_classes=9, pad_value=-999):
        super(ECGTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        #self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerPreLN(d_model, nhead, pad_value=pad_value) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, num_classes)
        self.pad_value = pad_value
        self.d_model=d_model
    def forward(self, x, mask):
        if torch.isnan(x).any():
            print("NaN in X")
            return
           
        batch_size, seq_len ,channels= x.shape 

        mask_values= x!=self.pad_value
        mask_values = mask_values.repeat(1, 1, (self.d_model // 3) + 1)  # Ripeti fino a 129
        mask_values = mask_values[..., :self.d_model]  # Tronca a 128
        
        x_proj = self.input_proj(x)
        if torch.isnan(x_proj).any():   
            print("NaN in proj")
            return
        
        #x = self.pos_encoder(x_proj, mask_values)
        x=x_proj* mask_values
        if torch.isnan(x).any():
            print("NaN in posEncoder")
            return

        for layer_idx, layer in enumerate(self.transformer_layers):
            x = layer(x,mask_values, mask,0)
            if torch.isnan(x).any():
                print("NaN in attention")
                return

        x_sum = (x * mask_values).sum(dim=1)  # Sum only valid tokens

        # Compute mean by dividing by the number of valid tokens
        x_mean = x_sum / mask_values.sum(dim=1).clamp(min=1)  # Avoid division by zero
        if torch.isnan(x_mean).any():
            print("NaN in media")
            return
        pred = self.fc(x_mean)

        if torch.isnan(pred).any():
            print("NaN in pred")
            return
        
        return pred  









'''
Train part
'''       

def print_memory_usage(stage):
    allocated = torch.cuda.memory_allocated() / 1024**2  # Converti in MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # Converti in MB
    print(f"🔹 {stage} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stampa il nome della GPU
gpu_name = torch.cuda.get_device_name(device)
#print(f"Stai usando la GPU: {gpu_name}")

# Inizializza il modello
model = ECGTransformer(num_classes=9).to(device)
if torch.cuda.device_count() > 1:
    #print(f"Usando {torch.cuda.device_count()} GPU!")
    model = torch.nn.DataParallel(model)

model.to(device)
# Sposta su GPU se disponibilex
#weights_tensor = weights_tensor.to(device)
# Loss, Ottimizzatore e Scheduler
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)


def train_model(model, train_loader, val_loader, epochs=50, patience=10):
    best_val_loss = float("inf")
    best_val_f1 = 0.0  # Miglior F1-score
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train, total_train = 0, 0
        
        for batch_idx, (inputs, masks, labels) in enumerate(train_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
        all_preds = []  # 🔹 Lista per salvare le predizioni
        all_labels = [] # 🔹 Lista per salvare le etichette reali
        
        with torch.no_grad():
            for batch_idx, (inputs, masks, labels) in enumerate(val_loader):
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                
                outputs = model(inputs, masks)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())   # 🔹 Salviamo le predizioni
                all_labels.extend(labels.cpu().numpy()) # 🔹 Salviamo le etichette reali
                
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val

        # 📌 Calcoliamo il Macro F1-score su tutte le classi
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        # Creazione della Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Stampa della Confusion Matrix in formato tabellare
        print("Confusion Matrix:")
        print(cm)
        scheduler.step()

        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {macro_f1:.4f}")

        # 📌 Early Stopping basato su F1-score invece della sola accuracy
        if val_loss < best_val_loss or macro_f1 > best_val_f1:
            best_val_loss = val_loss
            best_val_f1 = macro_f1
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
    model.eval()
    correct_chunks, total_chunks = 0, 0
    correct_signals, total_signals = 0, 0
    
    all_preds, all_labels, all_preds_chunks, all_labels_chunks = [], [], [], []
    print("inizio test")
    with torch.no_grad():
        for batch_idx,(inputs, masks, labels) in enumerate(test_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            num_chunks = inputs.shape[1]  # Numero di chunk nel batch
            signal_preds = []  # Predizioni per un singolo segnale
            #if batch_idx % 10 == 0:
                #print(f"batch numero: {batch_idx +1}")
            for i in range(num_chunks):
                chunk = inputs[:,i,:,:]  # Estraggo un chunk
                mask = masks[:,i,:]    # Estraggo la sua maschera
                output = model(chunk, mask)  # Passo il chunk nel modello
                pred = output.argmax(dim=1)  # Predizione del modello

                signal_preds.append(pred)  # Salvo la predizione
                # Accuratezza sul singolo chunk
                correct_chunks += (pred == labels).sum().item()
                total_chunks += labels.size(0)
            
            signal_preds_tensor = torch.stack(signal_preds)  # Shape: [299, 10]

            # Calcola la media lungo il primo asse (dim=0)
            final_signal_pred = signal_preds_tensor.float().mean(dim=0).round().long()  # Shape: [10]


            # Accuratezza sul segnale intero
            correct_signals += (final_signal_pred == labels).sum().item()
            total_signals += labels.size(0)

            all_preds.append(final_signal_pred)
            all_labels.append(labels)

    chunk_acc = correct_chunks / total_chunks if total_chunks > 0 else 0
    signal_acc = correct_signals / total_signals if total_signals > 0 else 0

    print(f"Chunk Accuracy: {chunk_acc:.4f}")
    print(f"Signal Accuracy: {signal_acc:.4f}")

    return all_preds, all_labels, chunk_acc, signal_acc

# 📌 Esegui il test
test_preds, test_labels, _,_ = test_model(model, test_loader)
print(test_labels)
print(test_preds)
true_labels = torch.cat(test_labels).cpu().numpy()
pred_labels = torch.cat(test_preds).cpu().numpy()

# Creazione della Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

# Stampa della Confusion Matrix in formato tabellare
print("Confusion Matrix:")
print(cm)

# Calcolo di Precision, Recall e F1-score
report = classification_report(true_labels, pred_labels, digits=4)

print("\nClassification Report:")
print(report)
