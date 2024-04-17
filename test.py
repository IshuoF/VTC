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