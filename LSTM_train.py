import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
from tqdm import tqdm
from VolleyballDataset import VolleyballDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])  # 只取最後一個時間步的輸出
        return out


def main():    

    max_length = 400
    transform = None
    train_set = VolleyballDataset(f'./dataset/train/train_last{max_length}_d.pkl', transform)

    valid_set = VolleyballDataset(f'./dataset/valid/valid_last{max_length}_d.pkl', transform)

    print(f"Train set length: {len(train_set)}")
    print(len(train_set[0]['position']))
    # find the class weights
    class_weights = [0] * 8

    for i in range(len(train_set)):
        label = int(train_set[i]['label'])
        class_weights[label] += 1

    print("Class number:")
    for i in range(8):
        print("label:", i, " count:", class_weights[i])

    class_weights = [0] * 8
    for i in range(len(valid_set)):
        label = int(valid_set[i]['label'])
        class_weights[label] += 1

    print("valid Class numbers:")
    for i in range(8):
        print("label:", i, " count:", class_weights[i])

    # # Check if GPU is available 

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    input_dim = 3    
    hidden_dim = 256
    layer_dim = 3
    output_dim = 8

    lr = 0.001
    num_epochs = 400
    batch_size = 32
    best_acc = 0
    patience, trials = 80, 0
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=4)
    
    train_losses = []
    valid_losses = []
    accuracies = []
    save_path = f'./saved_models/LSTM/ml{max_length}_bs{batch_size}_ep{num_epochs}_lr{lr}_test2'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(tqdm_train_loader):
            # Move data to device
            labels = batch["label"]
            positions = batch["position"]
            velocitys = batch["v"]
            timestamps = batch["t"] 
            original_len = batch["original_len"]
            
            labels, positions, velocitys, timestamps,original_len = labels.to(device), positions.to(device), velocitys.to(device), timestamps.to(device),original_len.to(device)
            
            velocitys = velocitys.unsqueeze(2)
            timestamps = timestamps.unsqueeze(2)
            original_len = original_len.unsqueeze(2)
            features = torch.cat((positions, original_len), dim=2)
            
           
            optimizer.zero_grad()
            output = model(positions)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        print(f'Epoch: {epoch+1:3d}. Loss: {train_loss:.4f}. ')
        # validation
        model.eval()
        valid_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                labels = batch["label"]
                positions = batch["position"]
                velocitys = batch["v"]
                timestamps = batch["t"] 
                original_len = batch["original_len"]
                
                labels, positions, velocitys, timestamps,original_len = labels.to(device), positions.to(device), velocitys.to(device), timestamps.to(device),original_len.to(device)
                velocitys = velocitys.unsqueeze(2)
                timestamps = timestamps.unsqueeze(2)
                original_len = original_len.unsqueeze(2)
                features = torch.cat((positions, original_len), dim=2)
                
                output = model(positions)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        
        acc = correct / total
        accuracies.append(acc)
        
        if epoch+1 % 5 == 0:
            print(f'Epoch: {epoch+1:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')
        if acc > best_acc:
            trials = 0
            best_acc = acc
            best_save_path = f'{save_path}/best.pth'
            torch.save(model.state_dict(), best_save_path)
            print(f'Epoch {epoch+1} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch+1}')
                break
        
    plt.figure(figsize=(10, 5))    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{save_path}/loss.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(' Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/accuracy.png')
            
if __name__ == "__main__":
    main()
   