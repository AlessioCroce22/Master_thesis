import zipfile
import os
from collections import defaultdict

def analizza_icentia_zip(zip_path, output_path="report_icentia.txt"):
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

# Esempio di utilizzo:
analizza_icentia_zip("C:/Users/aless/Downloads/icentia11k.zip")
