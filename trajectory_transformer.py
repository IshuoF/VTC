import torch
import torch.nn as nn
import torch.optim as optim
from VolleyballDataset import VolleyballDataset
from torch.utils.data import DataLoader,SequentialSampler
import numpy as np
import os

import argparse
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt



class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(torch.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src
    
class TrajEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TrajEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
    
    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
 

    
class Trajectory_Classifier(nn.Module):
    def __init__(self, encoder,d_model, num_classes, dropout=0.1):
        super(Trajectory_Classifier, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(d_model, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.linear = nn.Linear(128, num_classes)
    
    def forward(self, src):
        src = self.encoder(src)
        src = src.mean(dim=1)
        src = F.relu(self.dropout1(self.fc1(src)))
        src = F.relu(self.dropout2(self.fc2(src)))
        output = self.linear(src)
        return output




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="epoch num")
    parser.add_argument("--batch_size", default=16, type=int,  help="batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="num workers")
    parser.add_argument("--max_length", default=500, type=int, help="max sequence length")
    parser.add_argument("--lr", default=0.0008, type=float, help="learning rate")
    parser.add_argument("--save_dir", default="./saved_models", type=str, help="save directory")
    parser.add_argument("--exp_name", default="exp1", type=str, help="experiment name")
    args = parser.parse_args()
    best_loss = float('inf')
    max_length = args.max_length
    save_folder = str(max_length) + "_"+ args.exp_name
    save_dir = os.path.join(args.save_dir, save_folder)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Define the dataset
    transform = None
    train_set = VolleyballDataset(f'./dataset/train/train_last{max_length}.pkl', transform)
    valid_set = VolleyballDataset(f'./dataset/valid/valid_last{max_length}.pkl', transform)
   
    # print(train_set[0])
   
    # Check if GPU is available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")
        
    # Define the model
    d_model = 3
    nhead = 3
    dim_feedforward = 512
    num_layers = 6
    class_num = 8
    encoder = TrajEncoder(num_layers, d_model, nhead, dim_feedforward)
    model = Trajectory_Classifier(encoder, d_model, class_num)
    model.to(device)

    # Define the optimizer and loss function
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    
    # Define the dataloaders
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=4)
    
    losses = []
    valid_losses = []
    training_start_time = datetime.now()
    
    

    #Train the model
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(tqdm_train_loader):
            # Move data to device
            labels = batch["label"]
            positions = batch["position"]
            labels, positions = labels.to(device), positions.to(device)
            optimizer.zero_grad()
            output = model(positions)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / (i + 1)
        losses.append(average_loss) 

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}")   
        
        model.eval()
        total_valid_loss = 0.0
        correct = 0
        total = 0
        
        tqdm_valid_loader = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        with torch.no_grad():
            for i, batch in enumerate(tqdm_valid_loader ):
                labels = batch["label"]
                positions = batch["position"]
                labels, positions = labels.to(device), positions.to(device)
                output = model(positions)
                loss = criterion(output, labels)
                total_valid_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                print(predicted)
                print(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss
        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_save_path = os.path.join(save_dir, 'classifier_best.pth')
            best_save_path = best_save_path.replace("\\", "/")
            torch.save(model.state_dict(), best_save_path)

        last_save_path = os.path.join(save_dir, 'classifier_last.pth')
        last_save_path = last_save_path.replace("\\", "/")
        torch.save(model.state_dict(), last_save_path)
        
        accuracy = correct / total
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}, Validation Loss: {avg_valid_loss}, Validation Accuracy: {accuracy}")
    
    training_end_time = datetime.now()
    execution_time = training_end_time - training_start_time  
    formatted_time = str(execution_time) 
    
    print(f'Training Finish ! Best Loss is {best_loss:.5f}, Time Cost {formatted_time} s')
    
    
    plt.plot(losses, label='Training Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    save_path = save_path.replace("\\", "/")
    save_path = os.path.join(save_dir, 'training_loss_plot.png')
    plt.savefig(save_path)
    
    # model.eval()
    # all_labels = []
    # all_predictions = []
    # tqdm_test_loader = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    # with torch.no_grad():
    #     for i,batch in enumerate(tqdm_test_loader):
    #         labels = batch["label"]
    #         positions = batch["position"]
    #         labels, positions = labels.to(device), positions.to(device)
    #         output = model(positions)

    #         _, predicted = torch.max(output.data, 1)

    #         all_labels.extend(labels.cpu().numpy())
    #         all_predictions.extend(predicted.cpu().numpy())

    # # Convert lists to numpy arrays
    # all_labels = np.array(all_labels)
    # all_predictions = np.array(all_predictions)

    # # Calculate classification report
    # class_report = classification_report(all_labels, all_predictions)
    # print("Classification Report:\n", class_report)
    
    # del train_loader, test_loader, valid_loader, model, optimizer, criterion
    # torch.cuda.empty_cache()
if __name__ == "__main__":
    main()