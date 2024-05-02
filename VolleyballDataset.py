import torch
import pickle
from torch.utils.data import Dataset


class VolleyballDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.data = self.load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, position, velocities, timestamp = self.data[idx]
        
        label = torch.as_tensor(label, dtype=torch.long)
        position = torch.as_tensor(position, dtype=torch.float32)
        velocity= torch.as_tensor(velocities, dtype=torch.float32)
        timestamp = torch.as_tensor(timestamp, dtype=torch.float32)
        
        sample = {"label": label, "position": position, "v": velocity, "t": timestamp}
        
        if self.transform:
            position = self.transform(sample)
            
        return  sample
    
    def load_data(self):
        data = []
        
        with open(self.folder_path, 'rb') as f:
            data = pickle.load(f)

        return data
    
  