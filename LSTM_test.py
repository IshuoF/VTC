from LSTM_train import LSTMModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from VolleyballDataset import VolleyballDataset
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    max_length = 400
    testdataset_path = f"./dataset/test/test_last{max_length}_d.pkl"
    test_set = VolleyballDataset(testdataset_path, None)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
    
    pretrained_model_folder = f"./saved_models/LSTM/ml{max_length}_bs32_ep400_lr0.001_test2"
    pretrained_model_path = pretrained_model_folder+"/best.pth"
    
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

    input_dim = 3   
    hidden_dim = 256
    layer_dim = 3
    output_dim = 8

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
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
                velocitys = batch["v"]
                timestamps = batch["t"] 
            
                labels, positions, velocitys, timestamps= labels.to(device), positions.to(device), velocitys.to(device), timestamps.to(device)
                velocitys = velocitys.unsqueeze(2)
                timestamps = timestamps.unsqueeze(2)
                features = torch.cat((positions, velocitys,timestamps), dim=2)
                
                output = model(positions)
                _, predicted = torch.max(output.data, 1)
                    
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    print('All Predicted:', all_predicted)
    print('All Labels:', all_labels)
    
    
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=[str(i) for i in range(8)], 
                yticklabels=[str(i) for i in range(8)])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(pretrained_model_folder+'/test_confusion_matrix.png')
    # plt.show()
    
                
    classification_rep = classification_report(all_labels, all_predicted)  
    print('Classification Report:\n', classification_rep)  
    
    classification_data = []
    lines = classification_rep.split('\n')
    for line in lines[2:-4]:  # Skip first 2 lines and last 5 lines
        line_data = line.split()
        if len(line_data) == 0:
            continue
        row = {
            "class": int(line_data[0]),
            "precision": float(line_data[1]),
            "recall": float(line_data[2]),
            "f1-score": float(line_data[3]),
            "support": int(line_data[4])
        }
        classification_data.append(row)

    # Create DataFrame
    df_classification = pd.DataFrame(classification_data)

    # Save to CSV
    csv_filename = pretrained_model_folder+'/test_classification_report.csv'
    df_classification.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    main()