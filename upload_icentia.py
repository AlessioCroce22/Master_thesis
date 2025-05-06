import os
import math
import paramiko
from tqdm import tqdm

# === CONFIGURAZIONI ===

ZIP_FILE = os.path.expanduser("~/Downloads/icentia11k.zip")  # Cambia qui
CHUNK_SIZE_GB = 5  # Dimensione massima di ogni parte in GB
REMOTE_FOLDER = "/mnt/ssd1/croce"  # Cartella remota sul server

# Dati SFTP
SFTP_HOST = "192.168.110.41"
SFTP_PORT = 22
SFTP_USERNAME = "croce"
SFTP_PASSWORD = "croce"  # O usa autenticazione con chiave privata


def connect_sftp():
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    return paramiko.SFTPClient.from_transport(transport)


def split_and_upload_sftp(zip_path, chunk_size_gb, sftp, remote_folder):
    chunk_size = chunk_size_gb * 1024 * 1024 * 1024  # In byte
    total_size = os.path.getsize(zip_path)
    total_parts = math.ceil(total_size / chunk_size)

    # Ottieni lista dei file già presenti sul server
    try:
        remote_files = sftp.listdir(remote_folder)
    except FileNotFoundError:
        print(f"[!] La cartella remota '{remote_folder}' non esiste. Creala prima.")
        return

    with open(zip_path, 'rb') as f:
        for i in tqdm(range(total_parts), desc="Split & SFTP Upload"):
            part_filename = f"{os.path.basename(zip_path)}.part{i:03d}"

            if part_filename in remote_files:
                print(f"[SKIP] Parte {i:03d} già presente, si salta.")
                # Salta il blocco in lettura del file
                f.seek(chunk_size, 1)
                continue

            print(f"[UPLOAD] Parte {i:03d} in corso...")

            with open(part_filename, 'wb') as part_file:
                part_file.write(f.read(chunk_size))

            remote_path = f"{remote_folder.rstrip('/')}/{part_filename}"

            try:
                sftp.put(part_filename, remote_path)
                os.remove(part_filename)
                print(f"[✓] Parte {i:03d} caricata e rimossa localmente.")
            except Exception as e:
                print(f"[!] Errore upload parte {i:03d}: {e}")
                break

def report_upload_sftp(sftp, remote_folder):

        print(f"[UPLOAD] report in corso...")

        part_filename="report_icentia.txt"

        remote_path = f"{remote_folder.rstrip('/')}/{part_filename}"

        try:
            sftp.put(part_filename, remote_path)
            print(f"[✓] report caricato ")
        except Exception as e:
            print(f"[!] Errore upload report: {e}")


def main():
    if not os.path.exists(ZIP_FILE):
        print("File ZIP non trovato.")
        return

    print("Connessione SFTP in corso...")
    sftp = connect_sftp()
    print("Connesso a SFTP.")

    #split_and_upload_sftp(ZIP_FILE, CHUNK_SIZE_GB, sftp, REMOTE_FOLDER)
    report_upload_sftp(sftp,REMOTE_FOLDER)
    sftp.close()
    print("Upload completato e connessione chiusa.")


if __name__ == "__main__":
    main()
