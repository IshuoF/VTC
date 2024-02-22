import torch
import json
import os
from torch.utils.data import Dataset

class VolleyballDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.data = self.load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, position = self.data[idx]
        
        label = torch.as_tensor(label, dtype=torch.long)
        position = torch.as_tensor(position, dtype=torch.float32)
        
        if self.transform:
            position = self.transform(position)

        return  label, position
    
    def load_data(self):
        data = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(self.folder_path, filename)
                with open(json_file_path, 'r') as f:
                    trajectory_data = json.load(f)
                    
                    if trajectory_data.get("label") is None:
                        continue
                    
                    label = int(trajectory_data["label"])
                    positions = trajectory_data["frame_pos3d"]
                    frame_ids = list(map(int, positions.keys()))

                    
                    all_positions = [positions[str(frame_id)] for frame_id in frame_ids]
                    data.append((label, all_positions))  

        return data
    

if __name__ == "__main__":
    trajectory_path = "./data/trajectory_data/0204_man"
    transform = None
    volleyball_dataset = VolleyballDataset(trajectory_path, transform)
    
    sample_label, sample_position = volleyball_dataset[60]
    print("Label:", sample_label)
    print("Position:", sample_position)  # Removed "Transformed" from the print statement
