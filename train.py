import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from VolleyballDataset import VolleyballDataset
from models.Trajectory_Classifier import Trajectory_Classifier


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
    
        
    # # Define the model
    class_num = 8
    
    model = Trajectory_Classifier(d_model=3,
                                  dim_feedforward=64,
                                  n_layers=1,
                                  nhead=3,
                                  hidden_d=6,
                                  n_classes = 8,
                                  max_len=max_length,
                                  dropout=0.1)
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
    best_loss = float('inf')
    

    #Train the model
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(tqdm_train_loader):
            # Move data to device
            labels = batch["label"]
            
            positions = batch["position"] 
           
            labels, positions= labels.to(device), positions.to(device)
            optimizer.zero_grad()
            output = model(positions)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / (i + 1)
        losses.append(average_loss) 

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}")   
        
        
        ## validation
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
                
                print("output data",output.data)
                _, predicted = torch.max(output.data, 1)
                # print("predicted ",predicted)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                
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
    
    del train_loader, valid_loader, model, optimizer, criterion
    torch.cuda.empty_cache()
    


if __name__ == "__main__":
    main()
  