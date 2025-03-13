import wfdb
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split

#LOG 

# Funzione per ottenere le etichette HF etiology
def get_hf_etiology(csv_file, ids):
    processed_ids = [id[:-2] for id in ids]  # Rimuove gli ultimi due caratteri dagli ID
    
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        data = {row["Patient ID"]: row["HF etiology - Diagnosis"] for row in reader}

    return [data.get(pid, None) for pid in processed_ids]

# Funzione per caricare i segnali ECG
def load_ecg_data(patient_id, folder_path):
    file_path = os.path.join(folder_path, patient_id)
    record = wfdb.rdrecord(file_path)
    return record.p_signal  # Matrice (lunghezza, 3)

# Funzione per dividere i dati in train e test
def train_test(data_folder):
    patient_ids = [f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".dat")]
    return train_test_split(patient_ids, test_size=0.2, random_state=42)

# Funzione per calcolare la normalizzazione logaritmica
def compute_log_params(patient_ids, data_folder):
    eps = 1e-10  # Per evitare log(0)
    means = []
    stds = []
    
    for pid in patient_ids:
        signal = load_ecg_data(pid, data_folder)
        log_signal = np.log(np.abs(signal) + eps)  # LOG normalizzazione per canale
        means.append(np.mean(log_signal, axis=0))
        stds.append(np.std(log_signal, axis=0))
    
    return np.mean(means, axis=0), np.mean(stds, axis=0)

# Applica normalizzazione LOG
def log_normalization(signal, mean_log, std_log):
    eps = 1e-10
    log_signal = np.log(np.abs(signal) + eps)
    return (log_signal - mean_log) / std_log

# Funzione per il padding
def pad_signal(signal, target_length=1196000, pad_value=-999):
    current_length = signal.shape[0]
    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(signal, ((0, pad_width), (0, 0)), mode='constant', constant_values=pad_value)
    return signal

# Funzione per salvare i dati
def save_data_and_update_csv(signal, label, folder, prefix, index, csv_writer):
    file_name = f"{prefix}_{index}.npy"
    file_path = os.path.join(folder, file_name)
    
    np.save(file_path, signal)
    csv_writer.writerow([file_name, label])

# Cartelle dei dati
data_folder = r"/app/data/High-resolution_ECG"
train_folder = r"/app/work-data/Only-ECG-raw/train_log"
test_folder = r"/app/work-data/Only-ECG-raw/test_log"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

train_ids, test_ids = train_test(data_folder)

etichetta_train = get_hf_etiology(r"/app/data/subject-info.csv", train_ids)
etichetta_test = get_hf_etiology(r"/app/data/subject-info.csv", test_ids)

mean_log, std_log = compute_log_params(train_ids, data_folder)

# Salvataggio train
train_csv_path = os.path.join(train_folder, "train_labels.csv")
with open(train_csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["File Name", "Label"])
    
    for i, (pid, label) in enumerate(zip(train_ids, etichetta_train)):
        signal = load_ecg_data(pid, data_folder)
        signal = log_normalization(signal, mean_log, std_log)
        signal = pad_signal(signal)
        save_data_and_update_csv(signal, label, train_folder, "train", i, writer)

# Salvataggio test
test_csv_path = os.path.join(test_folder, "test_labels.csv")
with open(test_csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["File Name", "Label"])
    
    for i, (pid, label) in enumerate(zip(test_ids, etichetta_test)):
        signal = load_ecg_data(pid, data_folder)
        signal = log_normalization(signal, mean_log, std_log)
        signal = pad_signal(signal)
        save_data_and_update_csv(signal, label, test_folder, "test", i, writer)

print("✅ Tutti i file sono stati elaborati e salvati correttamente con normalizzazione LOG!")


'''
#MIN-MAX

# Funzione per ottenere le etichette HF etiology
def get_hf_etiology(csv_file, ids):
    hf_etiology_list = []
    processed_ids = [id[:-2] for id in ids]  # Rimuove gli ultimi due caratteri dagli ID
    
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        data = {row["Patient ID"]: row["HF etiology - Diagnosis"] for row in reader}

    return [data.get(pid, None) for pid in processed_ids]

# Funzione per caricare i segnali ECG
def load_ecg_data(patient_id, folder_path):
    file_path = os.path.join(folder_path, patient_id)
    record = wfdb.rdrecord(file_path)
    return record.p_signal  # Matrice (lunghezza, 3)

# Funzione per dividere i dati in train e test
def train_test(data_folder):
    patient_ids = [f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".dat")]
    return train_test_split(patient_ids, test_size=0.2, random_state=42)

# Funzione per normalizzare un singolo segnale con Min-Max Scaling (-1,1)
# Trova il min e max globale su tutto il training set
def compute_global_min_max(patient_ids, data_folder):
    min_val = np.inf
    max_val = -np.inf
    
    for pid in patient_ids:
        signal = load_ecg_data(pid, data_folder)
        min_val = np.minimum(min_val, np.min(signal, axis=0))  # Per canale
        max_val = np.maximum(max_val, np.max(signal, axis=0))  # Per canale
    
    return min_val, max_val

# Normalizza usando min e max globali
def min_max_normalization(signal, min_val, max_val):
    return 2 * (signal - min_val) / (max_val - min_val) - 1  # Scala tra -1 e 1

# Funzione per il padding
def pad_signal(signal, target_length=1196000, pad_value=-999):
    current_length = signal.shape[0]
    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(signal, ((0, pad_width), (0, 0)), mode='constant', constant_values=pad_value)
    return signal

# Funzione per salvare un file .npy e aggiornare il CSV
def save_data_and_update_csv(signal, label, folder, prefix, index, csv_writer):
    file_name = f"{prefix}_{index}.npy"
    file_path = os.path.join(folder, file_name)
    
    np.save(file_path, signal)  # Sovrascrive se esiste
    csv_writer.writerow([file_name, label])  # Scrive nel CSV

# Caricamento e preprocessing dei dati (senza tenere tutto in RAM)
data_folder = r"/app/data/High-resolution_ECG"
train_folder = r"/app/work-data/Only-ECG-raw/train_min_max"
test_folder = r"/app/work-data/Only-ECG-raw/test_min_max"

# Creazione cartelle se non esistono
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

train_ids, test_ids = train_test(data_folder)

etichetta_train = get_hf_etiology(r"/app/data/subject-info.csv", train_ids)
etichetta_test = get_hf_etiology(r"/app/data/subject-info.csv", test_ids)

global_min, global_max = compute_global_min_max(train_ids, data_folder)

# Salvataggio train
train_csv_path = os.path.join(train_folder, "train_labels.csv")
with open(train_csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["File Name", "Label"])
    
    for i, (pid, label) in enumerate(zip(train_ids, etichetta_train)):
        signal = load_ecg_data(pid, data_folder)
        signal = min_max_normalization(signal,global_min,global_max)
        signal = pad_signal(signal)
        save_data_and_update_csv(signal, label, train_folder, "train", i, writer)

# Salvataggio test
test_csv_path = os.path.join(test_folder, "test_labels.csv")
with open(test_csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["File Name", "Label"])
    
    for i, (pid, label) in enumerate(zip(test_ids, etichetta_test)):
        signal = load_ecg_data(pid, data_folder)
        signal = min_max_normalization(signal,global_min,global_max)
        signal = pad_signal(signal)
        save_data_and_update_csv(signal, label, test_folder, "test", i, writer)

print("✅ Tutti i file sono stati elaborati e salvati correttamente!")



#Z-SCORE

def get_hf_etiology(csv_file, ids):
    hf_etiology_list = []
    
    # Rimuove gli ultimi due caratteri da ogni ID
    processed_ids = [id[:-2] for id in ids]
    
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        data = {row["Patient ID"]: row["HF etiology - Diagnosis"] for row in reader}
    
    # Mantiene l'ordine originale degli ID
    for pid in processed_ids:
        hf_etiology_list.append(data.get(pid, None))
    
    return hf_etiology_list

# Funzione per leggere un file ECG
def load_ecg_data(patient_id, folder_path):
    file_path = os.path.join(folder_path, patient_id)
    record = wfdb.rdrecord(file_path)
    signal = record.p_signal  # Matrice (lunghezza, 3)
    return signal

def train_test(data_folder):
    # Percorso alla cartella con i file ECG
    # Lista di tutti i pazienti (ID unici dai file .dat)
    patient_ids = [f.split(".")[0] for f in os.listdir(data_folder) if f.endswith(".dat")]
    # Dividiamo in Train (80%) e Test (20%)
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    return train_ids, test_ids

# Funzione per calcolare la media e deviazione standard su train, canale per canale
def calculate_mean_std_on_train(train_data):
    mean = np.zeros(3)  # Tre canali
    std = np.zeros(3)
    n_samples = 0

    for signal in train_data:
        # Calcoliamo la media e la deviazione standard per ogni canale separatamente
        mean += np.mean(signal, axis=0)
        std += np.std(signal, axis=0)
        n_samples += signal.shape[0]

    mean /= n_samples
    std /= n_samples

    return mean, std

# Funzione di normalizzazione per ogni canale separatamente
def normalize_signal(signal, mean, std):
    return (signal - mean) / std

def pad_signal(signal, target_length=1196000, pad_value=-999):
    current_length = signal.shape[0]
    if current_length < target_length:
        pad_width = target_length - current_length
        padded_signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant', constant_values=pad_value)
        return padded_signal
    return signal


# Funzione per salvare i dati in formato .npy e creare i file CSV
def save_data_and_create_csv(train_data, test_data, etichetta_train, etichetta_test, train_folder, test_folder):
    # Salvataggio dei dati di train
    train_files = []
    for i, data in enumerate(train_data):
        file_name = f"train_{i}.npy"
        np.save(os.path.join(train_folder, file_name), data)
        train_files.append((file_name, etichetta_train[i]))

    # Salvataggio dei dati di test
    test_files = []
    for i, data in enumerate(test_data):
        file_name = f"test_{i}.npy"
        np.save(os.path.join(test_folder, file_name), data)
        test_files.append((file_name, etichetta_test[i]))

    # Creazione dei CSV per i dati di train e test
    with open(os.path.join(train_folder, "train_labels.csv"), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label"])  # Intestazione
        writer.writerows(train_files)

    with open(os.path.join(test_folder, "test_labels.csv"), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label"])  # Intestazione
        writer.writerows(test_files)


#split in train e test
data_folder=r"/app/data/High-resolution_ECG"
train_ids, test_ids = train_test(data_folder)

# Carichiamo i dati
train_data = [load_ecg_data(pid, data_folder) for pid in train_ids]
test_data = [load_ecg_data(pid, data_folder) for pid in test_ids]


#creazione liste di etichette
etichetta_train = get_hf_etiology(r"/app/data/subject-info.csv", train_ids)
etichetta_test = get_hf_etiology(r"/app/data/subject-info.csv", test_ids)


# Calcoliamo la media e la deviazione standard sui dati di train
mean, std = calculate_mean_std_on_train(train_data)

# Normalizziamo i dati (canale per canale)
train_data = [normalize_signal(sig, mean, std) for sig in train_data]
test_data = [normalize_signal(sig, mean, std) for sig in test_data]


# Applichiamo il padding per ottenere lunghezza uniforme
pad_value = -999  # Questo valore sarà usato per l'Attention Mask
train_data = [pad_signal(sig, pad_value=pad_value) for sig in train_data]
test_data = [pad_signal(sig, pad_value=pad_value) for sig in test_data]


# Directory dove salveremo i file .npy e i file CSV
train_folder = r"/app/work-data/Only-ECG-raw/train_Z"
test_folder = r"/app/work-data/Only-ECG-raw/test_Z"

# Creiamo le directory se non esistono
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Salva i dati e crea i CSV
save_data_and_create_csv(train_data, test_data, etichetta_train, etichetta_test, train_folder, test_folder)

print("tutti i file sono stati caricati")
'''