import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from VolleyballDataset import VolleyballDataset
from sklearn.metrics import confusion_matrix, classification_report


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
        src2 = self.linear2(torch.relu(self.linear1(src))) # Feedforward
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
    def __init__(self, encoder, d_model ,num_classes,max_length, dropout=0.1):
        super(Trajectory_Classifier, self).__init__()
        # [batch_size, seq_len, d_model]
        self.encoder = encoder  
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        # [d_model, num_classes]
        self.linear = nn.Linear(d_model, num_classes)
    
    def forward(self, src):
        B,L,D = src.size()
        class_token = self.class_token.expand(B, -1, -1)
        src = torch.cat((class_token, src), dim=1)
        src = self.encoder(src)
        src = self.norm(src)
        src = src[:, 0] # get the class token
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
    # Set the hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    max_length = args.max_length
    save_folder =  f'ml{max_length}_bs{batch_size}_ep{num_epochs}_lr{learning_rate}_{args.exp_name}'
    save_dir = os.path.join(args.save_dir, save_folder)
    
    # pretrained_model_path = "./saved_models/300_test3/classifier_best.pth"
     
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Define the dataset
    transform = None
    train_set = VolleyballDataset(f'./dataset/train/train_last{max_length}.pkl', transform)
    
    valid_set = VolleyballDataset(f'./dataset/valid/valid_last{max_length}.pkl', transform)
    
    
    # train_set = ConcatDataset([train_set, train_aug_set])
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
    
    # Check if GPU is available 
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
        
    # Define the model
    d_model = 3
    nhead = 3
    dim_feedforward = 512
    num_layers = 3
    class_num = 8
    encoder = TrajEncoder(num_layers, d_model, nhead, dim_feedforward)
    model = Trajectory_Classifier(encoder, d_model,class_num,max_length)
    # model.load_state_dict(torch.load(pretrained_model_path))
    model.to(device)

    # Define the optimizer and loss function
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Define the dataloaders
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=4)
    
   
    losses = []
    valid_losses = []
    training_start_time = datetime.now()
    total = 0
    total_correct = 0
    class_correct = [0] * class_num
    class_total = [0] * class_num
    best_loss = float('inf')
    all_predicted = []
    all_labels = []

    #Train the model
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(tqdm_train_loader):
            # Move data to device
            labels = batch["label"]
            # print("labels shape :",labels.shape)
            positions = batch["position"]
            # print("positions shape :",positions.shape)
            labels, positions = labels.to(device), positions.to(device)
            optimizer.zero_grad()
            output = model(positions)
            # print("output shape :",output.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / (i + 1)
        losses.append(average_loss) 

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}")   
        
        model.eval()
        total_valid_loss = 0.0
        
        
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
                print("predicted ",predicted)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                
                for c in range(class_num):
                    class_correct[c] += ((predicted == c) & (labels == c)).sum().item()
                    class_total[c] += (labels == c).sum().item()
                    
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            accuracy = total_correct / total
            print(f"Accuracy: {accuracy * 100:.2f}%")
            
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
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}, Validation Loss: {avg_valid_loss}")
    
    training_end_time = datetime.now()
    execution_time = training_end_time - training_start_time  
    formatted_time = str(execution_time) 
    # class_accuracies = [class_correct[c] / class_total[c] * 100 if class_total[c] != 0 else 0 for c in range(class_num)]
    
    # for c, acc in enumerate(class_accuracies):
    #     print(f"Class {c} Accuracy: {acc:.2f}%")
        
    print(f'Training Finish ! Best Loss is {best_loss:.5f}, Time Cost {formatted_time} s')
    
    # Plot the training loss
    
    plt.plot(losses, label='Training Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    loss_save_path = os.path.join(save_dir, 'training_loss_plot.png').replace("\\", "/")
    plt.savefig(loss_save_path)
    
    # Plot the accuracy matrix
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=range(class_num), yticklabels=range(class_num))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    acc_save_path = os.path.join(save_dir, 'accuracy_heatmap.png').replace("\\", "/")
    plt.savefig(acc_save_path)
    plt.savefig(acc_save_path)

    # Print and Save the classification report
    classification_report_result = classification_report(all_labels, all_predicted,output_dict=True)
    crr_df = pd.DataFrame(classification_report_result).transpose()
    crr_df.to_csv(os.path.join(save_dir, 'classification_report.csv').replace("\\", "/"), index=True)
    print("classificaiton report",classification_report(all_labels, all_predicted))
    
    del train_loader, valid_loader, model, optimizer, criterion
    torch.cuda.empty_cache()
    
def test():
    max_length = 1000
    testdataset_path = f"./dataset/test/test_last{max_length}.pkl"
    test_set = VolleyballDataset(testdataset_path, None)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
    
    pretrained_model_path = f"./saved_models/ml{max_length}_bs32_ep100_lr0.0001_dataset4922_layer3/classifier_best.pth"
    
    
    class_weights = [0] * 8
    for i in range(len(test_set)):
        label = int(test_set[i]['label'])
        class_weights[label] += 1

    print("Test Class numbers:")
    for i in range(8):
        print("label:", i, " count:", class_weights[i])
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
        
    # Define the model
    d_model = 3
    nhead = 3
    dim_feedforward = 512
    num_layers = 3
    class_num = 8
    encoder = TrajEncoder(num_layers, d_model, nhead, dim_feedforward)
    model = Trajectory_Classifier(encoder, d_model,class_num,max_length)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(device)
    model.eval()
    
    all_predicted = []
    all_labels = []
    tqdm_test_loader = tqdm(test_loader)
    with torch.no_grad():
            for i, batch in enumerate(tqdm_test_loader):
                labels = batch["label"]
                positions = batch["position"]
                labels, positions = labels.to(device), positions.to(device)
                output = model(positions)
                _, predicted = torch.max(output.data, 1)
                    
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
    classification_rep = classification_report(all_labels, all_predicted)  
    print('Classification Report:\n', classification_rep)   

if __name__ == "__main__":
    main()
    # test()