import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from .utils import convert_audio
import torch

class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate, channels):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_files = []
        
        for subdir, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith('.mp3'):
                    self.audio_files.append(os.path.join(subdir, file))
                elif file.lower().endswith('.wav'):
                    self.audio_files.append(os.path.join(subdir, file))
                elif file.lower().endswith('.au'):
                    self.audio_files.append(os.path.join(subdir, file))
                elif file.lower().endswith('.ogg'):
                    self.audio_files.append(os.path.join(subdir, file))
                elif file.lower().endswith('.flac'):
                    self.audio_files.append(os.path.join(subdir, file))
                elif file.lower().endswith('.aif'):
                    self.audio_files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            wav, sr = torchaudio.load(self.audio_files[idx])
            wav = convert_audio(wav, sr, self.sample_rate, self.channels)
            return self.audio_files[idx], wav
        except:
            return self.audio_files[idx], self.audio_files[idx]
