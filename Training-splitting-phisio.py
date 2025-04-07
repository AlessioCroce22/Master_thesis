import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_reference_data(reference_path):
    df = pd.read_csv(reference_path, header=None, names=["nomesegnale", "etichetta"])
    print("Label counts:")
    print(df["etichetta"].value_counts())
    return df.to_numpy()
def load_signals(signal_dir, reference_data):
    signals = {}
    for signal_name, _ in reference_data:
        mat_path = os.path.join(signal_dir, signal_name + ".mat")
        if os.path.exists(mat_path):
            mat_data = sio.loadmat(mat_path)
            for key in mat_data:
                if isinstance(mat_data[key], np.ndarray):
                    signals[signal_name] = np.squeeze(mat_data[key]) / 1000  # Normalize from 1000/mV
                    break
    return signals

def compute_metrics(signals):
    lengths = [len(sig) for sig in signals.values()]
    values = np.concatenate(list(signals.values()))
    
    metrics = {
        "mean": np.mean(values),
        "variance": np.var(values),
        "min": np.min(values),
        "max": np.max(values),
        "length_min": np.min(lengths),
        "length_max": np.max(lengths),
        "length_mean": np.mean(lengths)
    }
    return metrics

def normalize_signals(signals, mean, var):
    return {k: (v - mean) / np.sqrt(var) for k, v in signals.items()}

def resample_signals(signals, orig_freq=300, target_freq=250):
    return {k: signal.resample(v, int(len(v) * target_freq / orig_freq)) for k, v in signals.items()}

def pad_and_trim_signals(signals, target_length=60*250, pad_value=-100):
    adjusted_signals = {}
    for k, v in signals.items():
        if len(v) < target_length:
            adjusted = np.pad(v, (0, target_length - len(v)), constant_values=pad_value)
        else:
            adjusted = v[:target_length]
        adjusted_signals[k] = adjusted
    return adjusted_signals
def split_train_test(signals, reference_data, test_size=0.2):
    label_mapping = {label: idx for idx, label in enumerate(set(label for _, label in reference_data))}
    
    signal_names = list(signals.keys())
    labels = [label_mapping[label] for _, label in reference_data if _ in signals]
    
    train_signals, test_signals, train_labels, test_labels = train_test_split(
        signal_names, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    train_set = {name: np.append(signals[name], label) for name, label in zip(train_signals, train_labels)}
    test_set = {name: np.append(signals[name], label) for name, label in zip(test_signals, test_labels)}
    
    print("Training label distribution:")
    print(pd.Series(train_labels).value_counts())
    print("Test label distribution:")
    print(pd.Series(test_labels).value_counts())
    
    return train_set, test_set

def save_processed_signals(signals, reference_data, output_dir=r"C:\Users\aless\Downloads\phisionet_dataset\processed_signals"):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "training")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    label_mapping = {label: idx for idx, label in enumerate(set(label for _, label in reference_data))}
    
    signal_names = list(signals.keys())
    labels = [label_mapping[label] for _, label in reference_data if _ in signals]
    
    train_signals, test_signals, train_labels, test_labels = train_test_split(
        signal_names, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    train_label_counts = {label: 0 for label in set(train_labels)}
    test_label_counts = {label: 0 for label in set(test_labels)}
    
    for signal_name, label in zip(train_signals, train_labels):
        filename = os.path.join(train_dir, f"{signal_name}.npy")
        labeled_signal = np.append(signals[signal_name], label)  # Append numerical label
        np.save(filename, labeled_signal)
        train_label_counts[label] += 1
    
    for signal_name, label in zip(test_signals, test_labels):
        filename = os.path.join(test_dir, f"{signal_name}.npy")
        labeled_signal = np.append(signals[signal_name], label)  # Append numerical label
        np.save(filename, labeled_signal)
        test_label_counts[label] += 1
    
    print("Label distribution in saved training and test signals:")
    print("Training set:")
    for label, count in train_label_counts.items():
        print(f"  Label {label}: {count} signals")
    print("Test set:")
    for label, count in test_label_counts.items():
        print(f"  Label {label}: {count} signals")

def plot_random_signal(signals, title):
    random_key = np.random.choice(list(signals.keys()))
    plt.figure(figsize=(10, 4))
    plt.plot(signals[random_key])
    plt.title(f"{title}: {random_key}")
    plt.show()

def main():
    reference_path = r"C:\Users\aless\Downloads\phisionet_dataset\REFERENCE.csv"
    signal_dir = r"C:\Users\aless\Downloads\phisionet_dataset\signals"
    
    reference_data = load_reference_data(reference_path)
    signals = load_signals(signal_dir, reference_data)
    
    #print("Initial metrics:", compute_metrics(signals))
    #plot_random_signal(signals, "Original Signal")
    
    metrics = compute_metrics(signals)
    normalized_signals = normalize_signals(signals, metrics["mean"], metrics["variance"])
    #print("Metrics after normalization:", compute_metrics(normalized_signals))
    #plot_random_signal(normalized_signals, "Normalized Signal")
    
    resampled_signals = resample_signals(normalized_signals)
    #print("Metrics after resampling:", compute_metrics(resampled_signals))
    #plot_random_signal(resampled_signals, "Resampled Signal")
    
    final_signals = pad_and_trim_signals(resampled_signals)
    #print("Metrics after padding/trimming:", compute_metrics(final_signals))
    #plot_random_signal(final_signals, "Padded/Trimmed Signal")
    
    #labeled_signals_train,labeled_signals_test=split_train_test(final_signals, reference_data)
    #print("Train Metrics after labeling:", compute_metrics(labeled_signals_train))
    #print("Test Metrics after labeling:", compute_metrics(labeled_signals_test))
    #plot_random_signal(labeled_signals, "labeled Signal")
    save_processed_signals(final_signals,reference_data)
if __name__ == "__main__":
    main()
