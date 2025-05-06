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
from torch.utils.data import WeightedRandomSampler
import copy
from tqdm import tqdm
from torchvision.models import resnet18
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from transformers.models.longformer.modeling_longformer import LongformerConfig, LongformerSelfAttention
torch.backends.cudnn.benchmark = True
class SignalDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        signal = data[:-1]
        label = int(data[-1])

        # Mask e padding
        mask = (signal != -100).astype(np.float32)
        signal = signal * mask

        # AUGMENTATION solo se attivo e non tutto pad
        if self.augment and mask.sum() > 0:
            signal = self.apply_augmentation(signal, mask)

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def apply_augmentation(self, signal, mask):
        # Applica solo su parte valida
        valid = signal[mask == 1]

        # 1. Gaussian Noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, size=valid.shape)
            valid += noise

        # 2. Scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            valid *= scale

        # 3. Shifting
        if np.random.rand() < 0.4:
            shift = np.random.normal(0, 0.01)
            valid += shift

        # 4. Time Warping
        if np.random.rand() < 0.3:
            warp_factor = np.random.uniform(0.8, 1.2)
            warped = np.interp(
                np.linspace(0, len(valid), int(len(valid) * warp_factor)),
                np.arange(len(valid)),
                valid
            )
            if len(warped) < len(valid):
                valid = np.pad(warped, (0, len(valid) - len(warped)), mode='edge')
            else:
                valid = warped[:len(valid)]

        # 5. Trend Injection
        if np.random.rand() < 0.3:
            trend = np.linspace(0, np.random.uniform(-0.1, 0.1), num=len(valid))
            valid += trend

        # 6. Random Crop + Resample
        if np.random.rand() < 0.3:
            crop_size = int(len(valid) * np.random.uniform(0.8, 1.0))
            start = np.random.randint(0, len(valid) - crop_size)
            cropped = valid[start:start + crop_size]
            valid = np.interp(
                np.linspace(0, crop_size, len(valid)),
                np.arange(len(cropped)),
                cropped
            )

        signal[mask == 1] = valid
        return signal


def create_dataloaders(training_path, test_path, batch_size=32, seed=None):
    train_files = [os.path.join(training_path, f) for f in os.listdir(training_path) if f.endswith('.npy')]
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.npy')]
    
    total_size = len(train_files) + len(test_files)
    train_size = int(0.75 * total_size)
    val_size = int(0.05 * total_size)
    
    train_files, val_files = train_test_split(train_files, test_size=val_size, stratify=[np.load(f)[-1] for f in train_files], random_state=seed)
    
    # Recalcola le label dopo lo split
    train_labels = [np.load(f)[-1] for f in train_files]
    class_counts = Counter(train_labels)
    num_classes = len(class_counts)

    # Calcola pesi ammorbiditi per la loss
    #total = sum(class_counts.values())
    #weights = [((total / class_counts[i]) ** power) for i in range(num_classes)]
    #weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Sampler pesato per il train set
    #weights = [((total / class_counts[i]) ** 0.6) for i in range(num_classes)]
    #sample_weights = [weights[int(label)] for label in train_labels]
    #sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


    train_dataset = SignalDataset(train_files, True)
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


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000, downsample=4):
        super().__init__()
        effective_len = max_len // downsample
        pe = torch.zeros(effective_len, d_model)
        position = torch.arange(0, effective_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model, downsample=4):
        super().__init__()
        assert d_model % 3 == 0 and d_model % 2 == 0
        self.downsample = downsample

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, stride=downsample, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.extra_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.conv3_3  = nn.Conv1d(d_model, d_model // 3, kernel_size=3, padding=1)
        self.conv3_7  = nn.Conv1d(d_model, d_model // 3, kernel_size=7, padding=3)
        self.conv3_15 = nn.Conv1d(d_model, d_model // 3, kernel_size=15, padding=7)

        self.bn    = nn.BatchNorm1d(d_model)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.proj  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.extra_conv(x)
        residual = x
        x = torch.cat([self.conv3_3(x), self.conv3_7(x), self.conv3_15(x)], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.proj(residual)
        x = x.permute(0, 2, 1)
        return self.layer_norm(x)

# Residual Feedforward Classifier
class ResidualFC(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual
        return self.linear2(x)

# Transformer Encoder Layer with Longformer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size=512, dropout=0.1, max_len=15000, downsample=1):
        super().__init__()
        import math
        raw_len = max_len // downsample
        multiple = window_size * 2
        eff_seq_len = math.ceil(raw_len / multiple) * multiple

        config = LongformerConfig(
            attention_mode="sliding_chunks",
            hidden_size=d_model,
            num_attention_heads=nhead,
            num_hidden_layers=1,
            attention_window=[window_size],
            attention_dilation=[1],
            attention_dropout=dropout,
            hidden_dropout_prob=dropout,
            intermediate_size=4 * d_model,
            max_position_embeddings=eff_seq_len
        )
        self.longformer = LongformerModel(config)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, config.intermediate_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(config.intermediate_size, d_model)

    def forward(self, src, src_mask=None):
        hidden = self.norm1(src)
        seq_len = hidden.size(1)
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        out = self.longformer(
            inputs_embeds=hidden,
            attention_mask=src_mask.long() if src_mask is not None else None,
            position_ids=position_ids
        ).last_hidden_state
        src = src + self.dropout(out)
        ff = F.relu(self.linear1(self.norm2(src)))
        ff = self.linear2(self.dropout(ff))
        return src + self.dropout(ff)

# Final ECG Transformer
class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=96, num_heads=2, num_layers=4,
                 num_classes=4, dropout=0.2, pooling_type="hybrid",
                 max_len=15000, downsample=1, local_window=256):
        super().__init__()
        self.downsample = downsample
        self.local_window = local_window
        self.feature_extractor = CNNFeatureExtractor(d_model, downsample=downsample)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, downsample=downsample)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model, num_heads,
                window_size=local_window, dropout=dropout,
                max_len=max_len, downsample=downsample)
            for _ in range(num_layers)
        ])
        self.pooling_type = pooling_type
        self.attn_pool = nn.Linear(d_model, 1)
        self.fc = ResidualFC(d_model, num_classes, dropout=dropout)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.pos_encoding(x)

        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()

        # Downsample mask
        mask = mask.unsqueeze(1).float()
        mask = F.max_pool1d(mask, kernel_size=self.downsample, stride=self.downsample)
        mask = mask.squeeze(1).bool()

        # Pad to match Longformer requirement
        seq_len = x.size(1)
        multiple = self.local_window * 2
        pad_len = (multiple - seq_len % multiple) % multiple
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len), value=False)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_mask=mask)

        # Hybrid pooling
        attn_logits = self.attn_pool(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn_scores = F.softmax(attn_logits, dim=-1)
        attn_out = torch.sum(x * attn_scores.unsqueeze(-1), dim=1)

        mean_out = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.unsqueeze(-1).sum(dim=1).clamp(min=1)
        feat = (attn_out + mean_out) / 2
        return self.fc(feat)



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
    
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8,patience=5,min_lr=5e-5)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_val=200.0
    patience_counter = 0
    
    param_count=0
    for name, param in model.named_parameters():
        #print(name)
        param_count+=1
    print(f"learnable parameters: {param_count}")

    for epoch in range(num_epochs):
        start_time = time.time()
        #print(f"\n🟢 Epoch {epoch+1}/{num_epochs} - Training...")
        #train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, ncols=100)

        # ---------------- TRAINING --------------
        train_loss = 0.0
        all_preds, all_labels = [], []
        cont=0
        a=False
        for signals, masks, labels in train_loader:
            #if cont % 50 ==0:
            #print(f"batch numero: {cont+1}")
            signals, masks, labels = signals.cuda(device, non_blocking=True), masks.cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits = model(signals, masks)
            if torch.isnan(logits).any():
                a=True
                break
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    max_grad = param.grad.abs().max()
                    if torch.isnan(max_grad) or torch.isinf(max_grad):
                        print(f"loss: {loss}")
                        print(f"logits: {logits}")
                        print(f"labels: {labels}")
                        print(f"preds: {torch.argmax(logits, dim=1)}")
                        print(f"❌ NaN/Inf nei gradienti di {name}")
                        break
                    elif max_grad > 1e3:
                        print(f"⚠️ Gradiente molto alto ({max_grad.item():.2e}) in {name}")
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            cont+=1
        if a:
            break    
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        avg_train_loss = train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        #print(f"validation epoch {epoch+1}")
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
        scheduler.step(avg_val_loss)

        # Early Stopping check
        if avg_val_loss< best_val:
            best_val= avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "bestmodelloss.pt")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
        if val_f1>best_f1:
            best_f1=val_f1
            best_model_f1=copy.deepcopy(model.state_dict())
            torch.save(best_model_f1, "bestmodelf1.pt")
        if patience_counter >= patience:
            #print(f"Early stopping at epoch {epoch+1}")
            break

        # Stampa risultati
        elapsed_time = time.time() - start_time
        #print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed_time:.2f}s")
        #print(f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f}")
        #print(f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        #print(f"Class-wise F1: {class_f1}")
        #print(f"Confusion Matrix: {conf_matrix}")
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

    #print("\nTest Results:")
    print(f"Overall F1-score: {test_f1:.4f}")
    print(f"Class-wise F1: {class_f1}")


#print("inizio")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Esempio di utilizzo
training_path = "/app/work-data/training"
test_path = "/app/work-data/test"
batch_size = 8
# Controllo GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range (1,3):

    #seed= int(time.time()) % (2**32 - 1)
    #print(f"seed epoca numero {i+1}: {seed}")
    train_loader, val_loader, test_loader = create_dataloaders(training_path, test_path, batch_size, None)
    #verify_dataloaders(train_loader, val_loader)

    # Inizializza il modello
    model = ECGTransformer()

    # Addestramento
    best_model = train_model(model, train_loader, val_loader, num_epochs=300, lr=5e-4, patience=15)
    print(f"\ntest loss iterazione numero {i+1}: ")
    # Test finale
    model = setup_model(model)
    model.load_state_dict(torch.load("bestmodelloss.pt"))
    test_model(model, test_loader)
    print(f"\ntest f1 iterazione numero {i+1}: ")
    # Test finale
    model = setup_model(model)
    model.load_state_dict(torch.load("bestmodelf1.pt"))
    test_model(model, test_loader)
    i+=1

#tutte prove con un terzo di segnale dropout a 0.2, weight decay 1e-4 lr 1e-3 e linear attention
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

#stanotte fare 3 prove in parallelo con segnale intero: prima prova cnn pompata, pos encoding semplice, dropout 0.2, weight decay 1e-4 lr 5e-4 e trasformer leggero quindi 64,2,2: 48.95%
#seconda prova cnn potente pos encoding trasformer via di mezzo con 128,2,2 regolarizzazioni drop 0.3 weight 1e-4 lr 5e-4 : 50% 
#terza prova cnn potente, pos encoding, trasformer pompato regolarizzazioni drop 0.3 weight 1e-4 lr 5e-4 male pure questa


#ora su gpu 0 provo una nuova cnn e trasformer piu profondo ma meno complesso drop 0.3, 0.2 weight 1e-4: 60%
# su 2 uguale ma 128 dim e 4 head drop 0.4 0.3 weight 5e-4: 56%

#prova conf del 60% con una fc diversa e sampler a 0.6: 59%

#uguale ma senza sampler, 6 layer e con drop a 0.4 0.3: uguale

#ora riprovo col trasformer vecchio ossia linear attention:59%

#rimetto fc vecchia piu semplice e riduco ancora le capacita del tarsformer a dmodel 32 nhead 1 e num layer 4 drop 0.4,0.3: uguale

#prova con fc semplice cnn nuova (la quarta) gmodel 96 nhead 2 layer 4 drop 0.2 o 0.3 : 69.5%

#prova uguale ma con fc piu potente(la seconda) e layer 2 invece che 4: 67%

#ora provo uguale ma con 96 head 4 layer ma nhead 8: 65.6%

#prova uguale a 69.5 ma con 2 layer: 68.2%

#prova 69.5 ma con fc piu potente: 67.5%

#da provare anche la multihead attention e ad aumentare dmodel su entrambi. con linear e dmodel aumentato fa 67.9%

#ora provo trasformer lineare nhead 4 dmodel 96 layer 4 con cnn ultima e fc semplice ma dropout a 0.2 0.1 e attpool: 70%

#ora provo trasformer lineare nhead 4 dmodel 96 layer 4 con cnn ultima e fc semplice ma dropout a 0.2 0.1 ma con mean invece di attpool:68%

#prova uguale a gpu2 ma con learned pos encoding, parametri di scheduler diversi e drop a 0.3 su gpu0: 65.6%

#prova con pos encoding, tutto uguale ma layer 2 invece di 4 su gpu1: 63.21%

#uguale ma layer 6 su gpu2: 72.19% Class-wise F1: [0.70550162 0.86406844 0.67241379 0.64552661]

#lo provo con regolarizzazione forte forte drop 0.4 0.3 weight 2e-3 label smoothing 0.05: 68.36 ma da rifare magari prendendo f1 max per early stop[]: 

#ora provo con weight 1e-3: 67%

# su GPU 2 con nuovo scheduler, gpu1 solo senza smoothing e weight a 1e-3, gpu0 no smoothing, weight 1e-4 drop a 0.2 0.2 nessun miglioramento.

#su 2 data  fa 71.32%, su 0 uguale ma 3 head 66.77%, su 1 no aug ma fc nuova 68.8% ma F1 molto alte, ritentare con early stopping su f1


#data augemntation e nuovo trasformer prenorm invece che post, weight decay a 1e-4, dropout a 0.2 su fc e 0.1 nei layer, smoothing a 0.05 e lr a 1e-3 con reduceonplateau factor 0.5 patient 3 faccio 71%
#uguale ma con smoothing a 0.1 e drop a 0.2 e 0.3 fa 71.8%
#ora mantenendo questi ultimi parametri provo ad aumentare il trasformer dmodel 120 nhead 4 layer 3 e feed 768: 72.4%

#dopo excel provo uguale ma con media tra mean e attpool e dimezzo il downsampling multihead: va male, probabilmente è troppo poco potente per l input raddoppiato, il problema è la memoria che non basta

#provo multihead con residuo su ogni layer, residual fc finale, media tra meanpool e attpool: troppo tempo, riprovare

#ritorno con downsampling a /4 e multihead ma uso una CNN piu complessa. non cambia nulla

#provo trasformer diverso con longformer:
'''
d_model=96, num_heads=2, num_layers=4,
                 dropout=0.2, pooling_type="hybrid",
                 max_len=15000, downsample=1, local_window=256):
test loss iterazione numero 1: 
Overall F1-score: 0.7482
Class-wise F1: [0.70462633 0.89603025 0.66666667 0.72546858]

test f1 iterazione numero 1: 
Overall F1-score: 0.7370
Class-wise F1: [0.69257951 0.88772097 0.66019417 0.7073955 ]
'''
 
'''
blocco codice multihead attention e residual fc finale:

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3750):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_weights = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        attn_scores = self.attn_weights(x).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        return torch.sum(x * attn_weights, dim=1)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0 and d_model % 3 == 0

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.conv3_3 = nn.Conv1d(d_model, d_model // 3, kernel_size=3, padding=1)
        self.conv3_7 = nn.Conv1d(d_model, d_model // 3, kernel_size=7, padding=3)
        self.conv3_15 = nn.Conv1d(d_model, d_model // 3, kernel_size=15, padding=7)
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x
        x1 = self.conv3_3(x)
        x2 = self.conv3_7(x)
        x3 = self.conv3_15(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = x + self.proj(residual)
        x = x.permute(0, 2, 1)
        return self.layer_norm(x)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), key_padding_mask=src_mask)
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src

class ResidualFC(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual
        return self.linear2(x)

def masked_mean(x, mask):
    mask = mask.unsqueeze(-1)
    summed = (x * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)
    return summed / count

class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=96, num_heads=2, num_layers=4, num_classes=4, dropout=0.2, pooling_type="attention"):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model)

        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, num_heads, 768, 0.1)
            for _ in range(num_layers)
        ])

        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.pooling = AttentionPooling(d_model)
        else:
            self.pooling = None

        self.fc = ResidualFC(d_model, num_classes, dropout=dropout)

    def forward(self, x, mask):
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = self.pos_encoding(x)
        if torch.isnan(x).any():
            print("Nan in pos")
        mask = F.max_pool1d(mask.unsqueeze(1).float(), kernel_size=4, stride=4).squeeze(1).bool()
        attn_mask = ~mask

        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)
        if torch.isnan(x).any():
            print("Nan in layers")
        if self.pooling_type == "attention":
            x = self.pooling(x, mask)
        else:
            x = masked_mean(x, mask)
        if torch.isnan(x).any():
            print("Nan in attention")
        return self.fc(x)


test loss iterazione numero 1:                                                                                              
                                                                                                                            
Test Results:                                                                                                               
Overall F1-score: 0.7457                                                                                                    
Class-wise F1: [0.74637681 0.87607245 0.68292683 0.67759563]                                                                
                                                                                                                            
test f1 iterazione numero 1:                                                                                                
                                                                                                                            
Test Results:                                                                                                               
Overall F1-score: 0.7183                                                                                                    
Class-wise F1: [0.72435897 0.85534591 0.64912281 0.64417845]

test loss iterazione numero 2:                                                                                              
                                                                                                                            
Test Results:                                                                                                               
Overall F1-score: 0.7135                                                                                                    
Class-wise F1: [0.70989761 0.85848613 0.65546218 0.63001145]                                                                
                                                                                                                            
test f1 iterazione numero 2:                                                                                                

Test Results:                                                 
Overall F1-score: 0.7144                                      
Class-wise F1: [0.73376623 0.86300716 0.60162602 0.65914221]

test loss iterazione numero 3:                                

Test Results:                                                 
Overall F1-score: 0.7061                                      
Class-wise F1: [0.66909091 0.8666341  0.61538462 0.67342799]                                                                

test f1 iterazione numero 3:                                  

Test Results:                                                 
Overall F1-score: 0.7154                                      
Class-wise F1: [0.69480519 0.87014218 0.640625   0.65588915]

test loss iterazione numero 4:  

Test Results:
Overall F1-score: 0.7366
Class-wise F1: [0.70877193 0.86392252 0.69565217 0.67793031]

test f1 iterazione numero 4: 

Test Results:
Overall F1-score: 0.7122
Class-wise F1: [0.68608414 0.86714628 0.62962963 0.66593407]

test loss iterazione numero 5:                                                                                                                                          
                                                                                                                                                                        
Test Results:                                                                                                                                                           
Overall F1-score: 0.7022                                                                                                                                                
Class-wise F1: [0.70462633 0.85369387 0.60952381 0.6408377 ]                                                                                                            
                                                                                                                                                                        
test f1 iterazione numero 5:                                                                                                                                            
                                                                                                                                                                        
Test Results:                                                                                                                                                           
Overall F1-score: 0.6780                                                                                                                                                
Class-wise F1: [0.55411255 0.86520076 0.6557377  0.63702172]

test loss iterazione numero 6:                                                                                                                                          
                                                                                                                                                                        
Test Results:                                                                                                                                                           
Overall F1-score: 0.7273                                                                                                                                                
Class-wise F1: [0.72297297 0.85530864 0.64912281 0.68167861]                                                                                                            
                                                                                                                                                                        
test f1 iterazione numero 6:                                                                                                                                            
                                                                                                                                                                        
Test Results:                                                                                                                                                           
Overall F1-score: 0.7203
Class-wise F1: [0.7003367  0.86804901 0.66666667 0.64611872]

test loss iterazione numero 7: 

Test Results:
Overall F1-score: 0.7336
Class-wise F1: [0.74226804 0.85867446 0.67241379 0.6610703 ]

test f1 iterazione numero 7: 

Test Results:
Overall F1-score: 0.7315
Class-wise F1: [0.72909699 0.87074195 0.67272727 0.65348837]

test loss iterazione numero 8: 

Test Results:
Overall F1-score: 0.7262
Class-wise F1: [0.66666667 0.86765409 0.7008547  0.66948258]

test f1 iterazione numero 8: 

Test Results:
Overall F1-score: 0.7031
Class-wise F1: [0.6446281  0.8759542  0.61666667 0.67505241]

test loss iterazione numero 9: 
                                                                                    
Test Results:                                                                                                                                                           
Overall F1-score: 0.7232                                                                                                                                                
Class-wise F1: [0.72542373 0.85988484 0.64516129 0.66226623]
                                          
test f1 iterazione numero 9:   
                                                                                    
Test Results:                                                                                                                                                           
Overall F1-score: 0.7225                                                                                                                                                
Class-wise F1: [0.75167785 0.86279926 0.64150943 0.63397129]

test loss iterazione numero 10: 

Test Results:
Overall F1-score: 0.7087
Class-wise F1: [0.73611111 0.85810486 0.58252427 0.6581741 ]

test f1 iterazione numero 10: 

Test Results:
Overall F1-score: 0.7148
Class-wise F1: [0.69503546 0.8596577  0.62608696 0.67835052]

media su 10 iter 72.23%
'''






'''
terza cnn, con downsampling

class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3, stride=2),  # ↓ metà sequenza
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2, stride=2),  # ↓ ancora metà
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        self.layer_norm = nn.LayerNorm(d_model)  # LayerNorm per (B, T, D)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x  # residual dopo i primi due conv
        x = self.conv3(x)
        x = x + residual  # skip connection
        x = x.permute(0, 2, 1)  # (B, T, D) per LayerNorm
        x = self.layer_norm(x)
        return x



seconda cnn
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
'''

'''
modello con linea att custom:

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3750):
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
        self.self_attn = LinearAttention(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # PreNorm + residual attention
        normed_src = self.norm1(src)
        attn_output = self.self_attn(normed_src, mask=~src_mask) if src_mask is not None else self.self_attn(normed_src)
        src = src + self.dropout1(attn_output)

        # PreNorm + residual FF
        normed_src = self.norm2(src)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(normed_src))))
        src = src + self.dropout2(ff_output)

        return src



class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0 and d_model % 3 == 0, "d_model must be divisible by 6 (e.g., 60, 96, 120, etc.)"
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # Inception-style convolutions (each outputs d_model // 3 channels)
        self.conv3_3 = nn.Conv1d(d_model, d_model // 3, kernel_size=3, padding=1)
        self.conv3_7 = nn.Conv1d(d_model, d_model // 3, kernel_size=7, padding=3)
        self.conv3_15 = nn.Conv1d(d_model, d_model // 3, kernel_size=15, padding=7)

        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()

        # Projection to match residual (in case channels don't align perfectly)
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)  # Identity if same shape

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv1(x)               # (B, d_model//2, T/2)
        x = self.conv2(x)               # (B, d_model, T/4)
        residual = x                    # Save for residual connection

        # Inception block
        x1 = self.conv3_3(x)
        x2 = self.conv3_7(x)
        x3 = self.conv3_15(x)

        x = torch.cat([x1, x2, x3], dim=1)  # (B, d_model, T/4)
        x = self.bn(x)
        x = self.relu(x)

        x = x + self.proj(residual)    # Residual connection
        x = x.permute(0, 2, 1)         # (B, T, D) for LayerNorm
        x = self.layer_norm(x)
        return x


class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=108, num_heads=3, num_layers=5, num_classes=4, dropout=0.2, pooling_type="attention"):
        super().__init__()
        #prima cnn
        #self.feature_extractor = nn.Sequential(
        #    nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
        #    nn.ReLU(),
        #    nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2)
        #)
        self.feature_extractor = CNNFeatureExtractor(d_model)

        #self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        #self.pos_encoding = LearnedPositionalEncoding(max_len=3750, d_model=d_model)

        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, num_heads, 768, 0.1)
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
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, x, mask):
        batch_size, seq_len = x.shape
        #x = x.unsqueeze(-1)
        #x = self.proj(x)
        x = x.unsqueeze(1)  # (B, 1, T)
        if torch.isnan(x).any():
            print("signal contains nan ")
        x = self.feature_extractor(x)  # (B, d_model, T)
        #x = x.permute(0, 2, 1)  # (B, T, d_model)
        if torch.isnan(x).any():
            print("feature x contains nan ")
        x = self.pos_encoding(x)
        if torch.isnan(x).any():
            print("pos encoding x contains nan ")
        #if mask.sum(dim=1)==0:
            #print("Mask sums:", mask.sum(dim=1))  # dovrebbero essere tutte > 0
            #print("batch: ",x)
            #print("mask: ",mask)
        # Aggiorna la mask per la nuova lunghezza dopo il downsampling
        mask = F.max_pool1d(mask.unsqueeze(1).float(), kernel_size=4, stride=4).squeeze(1).bool()

        attn_mask = (mask == 0).to(torch.bool)
        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)

        if torch.isnan(x).any():
            print("trasformer contains nan")
        if self.pooling_type == "attention":
            x = self.pooling(x, mask)
        else:
            x = masked_mean(x, mask)
        if torch.isnan(x).any():
            print("pooling contains nan")
        logits=self.fc(x)
        if torch.isnan(logits).any():
            print("logits contains nan")
        return logits

test loss iterazione numero 1: 

Test Results:
Overall F1-score: 0.7159
Class-wise F1: [0.70289855 0.8583815  0.64661654 0.65587918]

test f1 iterazione numero 1: 

Test Results:
Overall F1-score: 0.7159
Class-wise F1: [0.70289855 0.8583815  0.64661654 0.65587918]

test loss iterazione numero 2: 

Test Results:
Overall F1-score: 0.7050
Class-wise F1: [0.70503597 0.86711281 0.59047619 0.65741729]

test f1 iterazione numero 2: 

Test Results:
Overall F1-score: 0.6956
Class-wise F1: [0.65789474 0.85628743 0.58064516 0.6877551 ]

test loss iterazione numero 3: 

Test Results:
Overall F1-score: 0.7045
Class-wise F1: [0.70149254 0.86116152 0.625      0.63033175]

test f1 iterazione numero 3: 

Test Results:
Overall F1-score: 0.7116
Class-wise F1: [0.7        0.85174419 0.63793103 0.65665236]

test loss iterazione numero 4:                                                                                                                                          
                                                                                                                                                                        
Test Results:                                                                                                                                                           
Overall F1-score: 0.7263                                                                                                                                                
Class-wise F1: [0.74021352 0.86520076 0.62745098 0.67235859]                                                                                                            
                                                                                                                                                                        
test f1 iterazione numero 4:                                                                                                                                            
                                                                                                                                                                        
Test Results:                                                                                                                                                           
Overall F1-score: 0.7042                                                                                                                                                
Class-wise F1: [0.69908815 0.86588235 0.62809917 0.62365591]

test loss iterazione numero 5:                                                                                                                        10:23:15 [26/1939]
                                                                                                                                                                        
Test Results:                                                                                                                                                           
Overall F1-score: 0.6990                                                                                                                                                
Class-wise F1: [0.67424242 0.86522782 0.59459459 0.66176471]                                                                                                            
                                                                                                                                                                        
test f1 iterazione numero 5:                                                                                                                                            
                                                                                                                                                                        
Test Results:
Overall F1-score: 0.7146
Class-wise F1: [0.70188679 0.8541563  0.63716814 0.66536585]
learnable parameters: 106
                                                                                                                                                                 
test loss iterazione numero 6: 

Test Results:
Overall F1-score: 0.7299
Class-wise F1: [0.68512111 0.86002886 0.72072072 0.65380493]

test f1 iterazione numero 6: 

Test Results:
Overall F1-score: 0.7244
Class-wise F1: [0.7254902  0.86717998 0.63768116 0.66741573]

test loss iterazione numero 7:                                                                                                                         19:07:07 [0/1939]

Test Results:
Overall F1-score: 0.6988
Class-wise F1: [0.73170732 0.85661253 0.56862745 0.63824885]

test f1 iterazione numero 7: 

Test Results:
Overall F1-score: 0.7127
Class-wise F1: [0.72312704 0.8540856  0.63157895 0.6419214 ]
learnable parameters: 106

test loss iterazione numero 8: 

Test Results:
Overall F1-score: 0.7010
Class-wise F1: [0.73646209 0.86578104 0.57142857 0.63046045]

test f1 iterazione numero 8: 

Test Results:
Overall F1-score: 0.6726
Class-wise F1: [0.60759494 0.8575419  0.61157025 0.61368653]

media  71.01%
'''


'''
#longformer unico
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000, downsample=4):
        super().__init__()
        effective_len = max_len // downsample
        pe = torch.zeros(effective_len, d_model)
        position = torch.arange(0, effective_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model, downsample=4):
        super().__init__()
        assert d_model % 3 == 0 and d_model % 2 == 0
        self.downsample = downsample

        # Use symmetric padding to preserve length when stride=1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, stride=downsample, padding=(7 - 1) // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=1, padding=(5 - 1) // 2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.conv3_3  = nn.Conv1d(d_model, d_model // 3,  kernel_size=3,  padding=(3 - 1) // 2)
        self.conv3_7  = nn.Conv1d(d_model, d_model // 3,  kernel_size=7,  padding=(7 - 1) // 2)
        self.conv3_15 = nn.Conv1d(d_model, d_model // 3, kernel_size=15, padding=(15 - 1) // 2)
        self.extra_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.bn    = nn.BatchNorm1d(d_model)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.proj  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv1(x)  # now preserves full length if downsample=1
        x = self.conv2(x)
        x = self.extra_conv(x)
        residual = x
        x1 = self.conv3_3(x)
        x2 = self.conv3_7(x)
        x3 = self.conv3_15(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.proj(residual)
        x = x.permute(0, 2, 1)
        return self.layer_norm(x)
class ResidualFC(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual
        return self.linear2(x)
class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=96, num_heads=2,
                 num_classes=4, dropout=0.2,
                 max_len=15000, downsample=1, local_window=768):
        super().__init__()
        self.downsample = downsample
        self.local_window = local_window

        self.feature_extractor = CNNFeatureExtractor(d_model, downsample=downsample)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, downsample=downsample)

        # One Longformer layer only
        padded_len = ((max_len + (local_window * 2 - 1)) // (local_window * 2)) * (local_window * 2)
        config = LongformerConfig(
            attention_window=local_window,
            hidden_size=d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=4,
            intermediate_size=4 * d_model,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=padded_len
        )
        self.longformer = LongformerModel(config)

        self.pooling_type = "hybrid"
        self.attn_pool = nn.Linear(d_model, 1)
        #self.fc = nn.Sequential(
        #    nn.Linear(d_model, d_model),
        #    nn.ReLU(),
        #    nn.Dropout(dropout),
        #    nn.Linear(d_model, num_classes)
        #)
        self.fc = ResidualFC(d_model, num_classes, dropout=dropout)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.feature_extractor(x)  # [B, L, D]
        x = self.pos_encoding(x)

        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()
        # Downsample mask
        mask = mask.unsqueeze(1).float()
        mask = F.max_pool1d(mask, kernel_size=self.downsample, stride=self.downsample)
        mask = mask.squeeze(1).bool()
        # Ensure sequence is padded to multiple of 2 * local_window for Longformer
        seq_len = x.size(1)
        required_multiple = self.local_window * 2
        pad_len = (required_multiple - seq_len % required_multiple) % required_multiple

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad time dimension
            mask = F.pad(mask, (0, pad_len), value=False)
        
        attention_mask = mask.long()  # 1 for valid, 0 for padding
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1  # First token gets global attention

        token_type_ids = torch.zeros_like(attention_mask)
        out = self.longformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state  # [B, L, D]

        # Mask-aware attention pooling
        attn_logits = self.attn_pool(out).squeeze(-1)  # [B, L]
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn_scores = F.softmax(attn_logits, dim=-1)  # [B, L]
        attn_out = torch.sum(out * attn_scores.unsqueeze(-1), dim=1)  # [B, D]
        mean_out = (out * mask.unsqueeze(-1).float()).sum(dim=1) / mask.unsqueeze(-1).sum(dim=1).clamp(min=1)
        feat = (attn_out + mean_out) / 2
        return self.fc(feat)

codice base
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000, downsample=4):
        super().__init__()
        effective_len = max_len // downsample
        pe = torch.zeros(effective_len, d_model)
        position = torch.arange(0, effective_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CNNFeatureExtractor(nn.Module):
    def __init__(self, d_model, downsample=4):
        super().__init__()
        assert d_model % 3 == 0 and d_model % 2 == 0
        self.downsample = downsample

        # Use symmetric padding to preserve length when stride=1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, stride=downsample, padding=(7 - 1) // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=1, padding=(5 - 1) // 2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.conv3_3  = nn.Conv1d(d_model, d_model // 3,  kernel_size=3,  padding=(3 - 1) // 2)
        self.conv3_7  = nn.Conv1d(d_model, d_model // 3,  kernel_size=7,  padding=(7 - 1) // 2)
        self.conv3_15 = nn.Conv1d(d_model, d_model // 3, kernel_size=15, padding=(15 - 1) // 2)
        self.extra_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.bn    = nn.BatchNorm1d(d_model)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.proj  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv1(x)  # now preserves full length if downsample=1
        x = self.conv2(x)
        x = self.extra_conv(x)
        residual = x
        x1 = self.conv3_3(x)
        x2 = self.conv3_7(x)
        x3 = self.conv3_15(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.proj(residual)
        x = x.permute(0, 2, 1)
        return self.layer_norm(x)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size=512, dropout=0.1, max_len=15000, downsample=4):
        super().__init__()
        # Compute padded length for positional embeddings
        import math
        raw_len = max_len // downsample
        multiple = window_size * 2
        eff_seq_len = math.ceil(raw_len / multiple) * multiple
        # Configure a single-layer Longformer for local attention
        config = LongformerConfig(
            attention_mode="sliding_chunks",
            hidden_size=d_model,
            num_attention_heads=nhead,
            num_hidden_layers=1,
            attention_window=[window_size],
            attention_dilation=[1],
            attention_dropout=dropout,
            hidden_dropout_prob=dropout,
            intermediate_size=4 * d_model,
            max_position_embeddings=eff_seq_len
        )
        from transformers.models.longformer.modeling_longformer import LongformerModel
        self.longformer = LongformerModel(config)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, config.intermediate_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(config.intermediate_size, d_model)

    def forward(self, src, src_mask=None):
        # src: [B, L, D], src_mask: [B, L] BoolTensor
        hidden = self.norm1(src)
        seq_len = hidden.size(1)
        # Prepare position ids for embeddings
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        # LongformerModel handles attention masking and position embeddings
        longformer_out = self.longformer(
            inputs_embeds=hidden,
            attention_mask=src_mask.long() if src_mask is not None else None,
            position_ids=position_ids
        ).last_hidden_state  # [B, L, D]
        src = src + self.dropout(longformer_out)
        ff = F.relu(self.linear1(self.norm2(src)))
        ff = self.linear2(self.dropout(ff))
        return src + self.dropout(ff)

class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=96, num_heads=2, num_layers=4,
                 num_classes=4, dropout=0.2, pooling_type="hybrid",
                 max_len=15000, downsample=1, local_window=512):
        super().__init__()
        self.downsample = downsample
        self.local_window = local_window
        self.feature_extractor = CNNFeatureExtractor(d_model, downsample=downsample)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, downsample=downsample)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model, num_heads,
                window_size=local_window, dropout=dropout,
                max_len=max_len, downsample=downsample)
            for i in range(num_layers)
        ])
        self.pooling_type = pooling_type
        self.attn_pool = nn.Linear(d_model, 1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask=None):
        # x: [B, seq_len] or [B, seq_len, C]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.pos_encoding(x)

        # Prepare mask: True=valid, False=pad
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()
        # Downsample mask
        mask = mask.unsqueeze(1).float()
        mask = F.max_pool1d(mask, kernel_size=self.downsample, stride=self.downsample)
        mask = mask.squeeze(1).bool()

        # Pad sequence & mask to multiple of local_window*2
        seq_len = x.size(1)
        multiple = self.local_window * 2
        if seq_len % multiple != 0:
            pad_len = multiple - (seq_len % multiple)
            x = F.pad(x, (0, 0, 0, pad_len))  # pad time dim
            mask = F.pad(mask, (0, pad_len), value=False)

        # Pass through Transformer layers with full mask
        for layer in self.layers:
            # src_mask passed as valid positions (True)
            x = layer(x, src_mask=mask)

        # Hybrid pooling
        attn_scores = F.softmax(self.attn_pool(x).squeeze(-1), dim=-1)
        attn_out = torch.sum(x * attn_scores.unsqueeze(-1), dim=1)
        mean_out = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.unsqueeze(-1).sum(dim=1).clamp(min=1)
        feat = (attn_out + mean_out) / 2
        return self.fc(feat)
'''