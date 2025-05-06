import os
import io
import zipfile
import numpy as np
import random
import torch
import time
from torch.utils.data import Dataset, DataLoader
import wfdb

# Mean and std precomputed
MEAN = 0.001829
STD = 1.371429

# Beat label mapping
ds_beat_names = {
    0: 'undefined',
    1: 'normal',
    2: 'pac',
    3: 'aberrated',
    4: 'pvc'
}
_HI_PRIO_BEATS = [2, 3, 4]
_LO_PRIO_BEATS = [0, 1]

def get_complete_beats(indices, labels=None, start=0, end=None):
    if end is None:
        end = indices[-1]
    if start >= end:
        raise ValueError('`end` must be greater than `start`')
    start_index = np.searchsorted(indices, start, side='left') + 1
    end_index = np.searchsorted(indices, end, side='right')
    indices_slice = indices[start_index:end_index]
    if labels is None:
        return indices_slice
    else:
        label_slice = labels[start_index:end_index]
        return indices_slice, label_slice

def get_beat_label(labels):
    beat_counts = np.bincount(labels, minlength=len(ds_beat_names))
    most_hp_beats = np.argmax(beat_counts[_HI_PRIO_BEATS])
    if beat_counts[_HI_PRIO_BEATS][most_hp_beats] > 0:
        y = _HI_PRIO_BEATS[most_hp_beats]
    else:
        most_lp_beats = np.argmax(beat_counts[_LO_PRIO_BEATS])
        y = _LO_PRIO_BEATS[most_lp_beats] if beat_counts[_LO_PRIO_BEATS][most_lp_beats] > 0 else 0
    return y

def _get_segments_from_zip(zip_file):
    all_files = zip_file.namelist()
    hea_files = [f for f in all_files if f.endswith('.hea') and '_s' in f]
    print(f"[INFO] Total .hea files found: {len(hea_files)}")
    return hea_files

def load_segment_from_zip(zip_file, base_path):
    base_path = base_path.replace('.hea', '')
    try:
        with zip_file.open(base_path + '.dat') as datf, \
             zip_file.open(base_path + '.atr') as atrf, \
             zip_file.open(base_path + '.hea') as heaf:

            with open("temp.dat", "wb") as temp_dat, open("temp.atr", "wb") as temp_atr, open("temp.hea", "wb") as temp_hea:
                temp_dat.write(datf.read())
                temp_atr.write(atrf.read())
                temp_hea.write(heaf.read())

            record = wfdb.rdrecord("temp", channels=[0])
            annotation = wfdb.rdann("temp", 'atr')

            signal = record.p_signal.flatten()
            beat_ends = annotation.sample
            beat_labels = annotation.symbol

            beat_label_map = {'N': 1, 'V': 4, 'A': 2, 'L': 3, '~': 0}
            beat_labels = np.array([beat_label_map.get(sym, 0) for sym in beat_labels])

            return signal, (beat_ends, beat_labels)

    except Exception as e:
        #print(f"[ERROR] Failed loading segment {base_path}: {e}")
        return None, (None, None)

class ECGFrameDataset(Dataset):
    def __init__(self, zip_path, frame_size=2048, samples_per_patient=4096):
        self.zip_path = zip_path
        self.frame_size = frame_size
        self.samples_per_patient = samples_per_patient
        self.indices = self._precompute_indices()

    def _precompute_indices(self):
        all_indices = []
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            segment_files = _get_segments_from_zip(zf)
            segment_groups = {}
            for f in segment_files:
                pid = f.split('/')[2]
                segment_groups.setdefault(pid, []).append(f)

            patient_ids = list(segment_groups.keys())
            random.shuffle(patient_ids)

            for pid in patient_ids:
                segments = segment_groups[pid]
                random.shuffle(segments)
                samples = 0
                while samples < self.samples_per_patient:
                    seg = random.choice(segments)
                    base_path = seg.replace('.hea', '')
                    try:
                        with zf.open(base_path + '.hea') as heaf:
                            lines = heaf.read().decode().splitlines()
                            sig_len = int(lines[0].split()[3])
                            if sig_len < self.frame_size:
                                continue
                            for _ in range(4):  # sample a few frames per segment
                                start = random.randint(0, sig_len - self.frame_size)
                                end = start + self.frame_size
                                all_indices.append((base_path, start, end))
                                samples += 1
                                if samples >= self.samples_per_patient:
                                    break
                    except:
                        continue
        print(f"[INFO] Total index tuples generated: {len(all_indices)}")
        return all_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_path, start, end = self.indices[idx]
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            signal, (beat_ends, beat_labels) = load_segment_from_zip(zf, base_path)
            if signal is not None:
                frame = signal[start:end]
                frame = np.nan_to_num(frame, nan=MEAN)
                frame = (frame - MEAN) / STD

                _, frame_beat_labels = get_complete_beats(beat_ends, beat_labels, start, end)
                label = get_beat_label(frame_beat_labels)

                signal_tensor = torch.tensor(frame, dtype=torch.float32)
                label_tensor = torch.tensor(label, dtype=torch.long)
                mask_tensor = torch.ones_like(signal_tensor, dtype=torch.float32)
                return signal_tensor, mask_tensor, label_tensor

def collate_fn(batch):
    signals, masks, labels = zip(*batch)
    return torch.stack(signals), torch.stack(masks), torch.stack(labels)

def get_dataloaders(zip_path, batch_size=32, frame_size=2048):
    dataset = ECGFrameDataset(zip_path, frame_size)
    dataset_len = len(dataset)
    split_idx = int(0.95 * dataset_len)
    train_dataset = torch.utils.data.Subset(dataset, list(range(split_idx)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(split_idx, dataset_len)))

    print(f"[INFO] Train dataset size: {len(train_dataset)}")
    print(f"[INFO] Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    return train_loader, val_loader

base_dir = "/app/data/icentia11k.zip"
batch_size = 32
start_time = time.time()
train_loader, val_loader = get_dataloaders(base_dir, batch_size)

# Sanity check
for i, (signals, masks, labels) in enumerate(train_loader):
    print(f"\nBatch {i+1}")
    print(f" Signal shape: {signals.shape}")
    print(f" Mask shape:   {masks.shape}")
    print(f" Label shape:  {labels.shape}")
    break
elapsed_time = time.time() - start_time
print(f"Time: {elapsed_time:.2f}s")