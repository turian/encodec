import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from .utils import convert_audio
import torch

class MP3Dataset(Dataset):
    def __init__(self, root_dir, sample_rate, channels):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.mp3_files = []
        
        for subdir, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith('.mp3'):
                    self.mp3_files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.mp3_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            wav, sr = torchaudio.load(self.mp3_files[idx])
            wav = convert_audio(wav, sr, self.sample_rate, self.channels)
            return self.mp3_files[idx], wav
        except:
            return self.mp3_files[idx], self.mp3_files[idx]
