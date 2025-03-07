# Usa un'immagine base di Python
FROM python:3.10

# Imposta la directory di lavoro dentro il container
WORKDIR /app

# Copia il file requirements.txt e installa le dipendenze
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice nel container
COPY . .

# Comando di default all'avvio del container
CMD ["python", "ProvaScript.py"]

