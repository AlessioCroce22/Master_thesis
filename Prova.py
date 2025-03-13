import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

# Funzione per caricare i segnali ECG
def load_ecg_signals(folder, num_files=100):
    files = [f.split(".")[0] for f in os.listdir(folder) if f.endswith(".dat")]
    signals = []
    
    for file in files[:num_files]:  # Prendi solo alcuni segnali per analisi rapida
        record = wfdb.rdrecord(os.path.join(folder, file))
        signals.append(record.p_signal)
    
    return signals

# Funzione per calcolare statistiche dei segnali
def compute_statistics(signal):
    stats = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "skewness": skew(signal.flatten()),
        "kurtosis": kurtosis(signal.flatten()),
        "outliers": np.sum((signal < np.percentile(signal, 1)) | (signal > np.percentile(signal, 99)))
    }
    return stats

# Funzione per applicare diverse normalizzazioni
def normalize_signals(signals, method):
    transformed_signals = []
    
    for signal in signals:
        new_signal = np.copy(signal)
        for i in range(signal.shape[1]):  # Canali
            ch = signal[:, i]
            if method == "z-score":
                scaler = StandardScaler()
            elif method == "min-max":
                scaler = MinMaxScaler(feature_range=(-1, 1))
            elif method == "robust":
                scaler = RobustScaler()
            elif method == "log":
                ch = np.log1p(np.abs(ch)) * np.sign(ch)  # Log con segno
                transformed_signals.append(new_signal)
                continue
            elif method == "power":
                scaler = PowerTransformer(method='yeo-johnson')
            else:
                raise ValueError("Metodo di normalizzazione non valido")
            
            new_signal[:, i] = scaler.fit_transform(ch.reshape(-1, 1)).flatten()
        transformed_signals.append(new_signal)
    
    return transformed_signals

# Funzione per valutare le normalizzazioni
def evaluate_normalizations(signals):
    methods = ["z-score", "min-max", "robust", "log", "power"]
    scores = {}
    
    for method in methods:
        transformed_signals = normalize_signals(signals, method)
        all_stats = [compute_statistics(sig) for sig in transformed_signals]
        
        # Calcoliamo le metriche per il confronto
        avg_outliers = np.mean([s["outliers"] for s in all_stats])
        avg_std = np.mean([s["std"] for s in all_stats])
        avg_skewness = np.mean([abs(s["skewness"]) for s in all_stats])  # Minore è meglio
        avg_kurtosis = np.mean([s["kurtosis"] for s in all_stats])  # Vicino a 3 è meglio
        
        scores[method] = {
            "outliers": avg_outliers,
            "std": avg_std,
            "skewness": avg_skewness,
            "kurtosis": avg_kurtosis
        }
    
    # Seleziona il metodo migliore con meno outlier e distribuzione più normale
    best_method = min(scores, key=lambda m: (scores[m]["outliers"], scores[m]["skewness"], abs(scores[m]["kurtosis"] - 3)))
    
    return scores, best_method

# Funzione per eseguire l'analisi completa
def analyze_signals(data_folder, num_files=5):
    signals = load_ecg_signals(data_folder, num_files)
    
    print("\n📊 Statistiche iniziali:")
    for i, signal in enumerate(signals):
        stats = compute_statistics(signal)
        print(f"🔹 Segnale {i}: {stats}")
    
    scores, best_method = evaluate_normalizations(signals)
    
    print("\n📈 Risultati Normalizzazione:")
    for method, stats in scores.items():
        print(f"{method.upper()} -> Outliers: {stats['outliers']:.2f}, Std: {stats['std']:.4f}, Skewness: {stats['skewness']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")
    
    print(f"\n🏆 Normalizzazione migliore: **{best_method.upper()}**")
    
    return best_method

# Esegui analisi
data_folder = r"C:\Users\aless\OneDrive\Desktop\Alessio\Alessio_magistrale\Tesi_magistrale\MUSIC-dataset\High-resolution_ECG"
best_norm = analyze_signals(data_folder, num_files=100)
