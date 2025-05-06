import io
import wfdb
import tempfile
import torch
from tqdm import tqdm
import zipfile
import os
from collections import defaultdict

def analizza_icentia_zip(zip_path, output_path="/app/data/report_icentia_lab.txt"):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        struttura = defaultdict(lambda: defaultdict(lambda: {'.dat': 0, '.atr': 0, '.hea': 0}))
        root_prefix = "icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0/"
        size_bytes = os.path.getsize(zip_path)
        size_mb = size_bytes / (1024 * 1024)
        for item in zipf.namelist():
            if not item.startswith(root_prefix) or item.endswith('/'):
                continue  # Skip directories or files outside root

            rel_path = item[len(root_prefix):]
            parts = rel_path.split('/')

            if len(parts) == 3:
                pxx, pxxyyy, filename = parts
                _, ext = os.path.splitext(filename)

                if ext in ['.dat', '.atr', '.hea']:
                    struttura[pxx][pxxyyy][ext] += 1

        total_atr = total_dat = total_hea = 0
        total_files_per_folder = []

        lines = []
        lines.append(f"Cartelle di primo livello (tipo pXX): {len(struttura)}\n")

        for pxx in sorted(struttura.keys()):
            subfolders = struttura[pxx]
            lines.append(f"\n {pxx}/ - {len(subfolders)} sottocartelle")
            for pxxyyy in sorted(subfolders.keys()):
                counts = subfolders[pxxyyy]
                dat = counts['.dat']
                atr = counts['.atr']
                hea = counts['.hea']
                total_dat += dat
                total_atr += atr
                total_hea += hea
                total = dat + atr + hea
                total_files_per_folder.append(total)
                lines.append(f" {pxxyyy}/ - .dat: {dat}, .atr: {atr}, .hea: {hea}  (totale: {total})")

        # Statistiche generali
        total_folders = len(total_files_per_folder)
        max_files = max(total_files_per_folder) if total_folders else 0
        min_files = min(total_files_per_folder) if total_folders else 0
        avg_files = sum(total_files_per_folder) / total_folders if total_folders else 0

        lines.append("\n\n Statistiche generali:")
        lines.append(f"Totale file .dat: {total_dat}")
        lines.append(f"Totale file .atr: {total_atr}")
        lines.append(f"Totale file .hea: {total_hea}")
        lines.append(f"Totale cartelle pXXYYY: {total_folders}")
        lines.append(f"Numero massimo di file in una cartella: {max_files}")
        lines.append(f"Numero minimo di file in una cartella: {min_files}")
        lines.append(f"Numero medio di file per cartella: {avg_files:.2f}")
        lines.append(f"Dimensione file ZIP: {size_mb:.2f} MB ({size_bytes} bytes)\n")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Report salvato in: {output_path}")
def confronta_file_txt(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        contenuto1 = f1.read()
        contenuto2 = f2.read()

    if contenuto1 == contenuto2:
        print("✅ I due file sono identici.")
        return True
    else:
        print("❌ I file sono diversi.")
        return False
def verify_dataset():
    analizza_icentia_zip("/app/data/icentia11k.zip")
    confronta_file_txt("/app/data/report_icentia_lab.txt", "/app/data/report_icentia.txt")
   
def calculate_mean_dev():
    zip_path = "/app/data/icentia11k.zip"  # Cambia path se necessario
    dataset_root = "icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0"

    # Variabili per media e deviazione su tutto il dataset
    total_sum = torch.tensor(0.0, dtype=torch.float64)
    total_sq_sum = torch.tensor(0.0, dtype=torch.float64)
    total_count = torch.tensor(0, dtype=torch.int64)


    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Trova tutti i file .hea (sono gli unici necessari per identificare i record)
        hea_files = [f for f in zf.namelist() if f.endswith(".hea") and f.startswith(dataset_root)]

        for hea_file in tqdm(hea_files, desc="Processing records"):
            base_path = hea_file[:-4]  # Rimuove ".hea"

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Estrai .hea, .dat, .atr
                    for ext in [".hea", ".dat", ".atr"]:
                        zf.extract(base_path + ext, path=tmpdir)
                    
                    # Percorso temporaneo del record
                    rec_dir = os.path.join(tmpdir, os.path.dirname(base_path))
                    rec_name = os.path.basename(base_path)

                    record = wfdb.rdrecord(os.path.join(rec_dir, rec_name))
                    signal = torch.from_numpy(record.p_signal).squeeze()  # (n_samples, 1) → (n_samples)
                    signal = torch.nan_to_num(signal, nan=0.0)
                    # Calcola media e varianza incrementale
                    total_sum += (signal).sum().item()
                    total_sq_sum += ( signal** 2).sum().item()
                    total_count += signal.numel()
                    if torch.isnan(torch.tensor(total_sum)).any() or torch.isnan(torch.tensor(total_sq_sum)).any() or torch.isnan(torch.tensor(total_count)).any():
                        print("NaN detected")
                        break
            except Exception as e:
                continue
                #print(f"⚠️ Errore su {base_path}: {e}")

    # Calcolo finale
    mean = total_sum / total_count
    std = ((total_sq_sum / total_count) - mean**2) ** 0.5

    print("\n✅ RISULTATI FINALI:")
    print(f"Media totale: {mean:.6f}")
    print(f"Deviazione standard totale: {std:.6f}")

    
#✅ RISULTATI FINALI:
#Media totale: 0.001829
#Deviazione standard totale: 1.371429