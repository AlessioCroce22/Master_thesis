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



class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=4):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # (B, C, 1)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# Per creare il modello:
def ECGResNet18_1D(in_channels=1, num_classes=4):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], in_channels, num_classes)




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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10)
    #scaler = torch.amp.GradScaler(device="cuda")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_val=200.0
    patience_counter = 0    
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
            signals, masks, labels = signals[:,2500:7500].cuda(device, non_blocking=True), masks[:,2500:7500].cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)
            
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
                signals, masks, labels = signals[:,2500:7500].cuda(device, non_blocking=True), masks[:,2500:7500].cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)
                
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
        scheduler.step(val_f1)

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
            signals, masks, labels = signals[:,2500:7500].cuda(device, non_blocking=True), masks[:,2500:7500].cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)

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
batch_size = 256
train_loader, val_loader, test_loader = create_dataloaders(training_path, test_path, batch_size)
#verify_dataloaders(train_loader, val_loader)


# Controllo GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Inizializza il modello
model = ECGResNet18_1D(1,4)

# Addestramento
best_model = train_model(model, train_loader, val_loader, num_epochs=200, lr=1e-3, patience=30)
print("test")
# Test finale
test_model(best_model, test_loader)



