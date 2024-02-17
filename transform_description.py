
import pandas as pd
import os
import chardet

folder_path = '.data/trajectory_text_dataset/origin'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]


def transform_description(description):
    replacement_mapping = {
        "an unreturnable spike":"The rally concludes with an unreturnable spike.",
        "The ball went out after the opponent's touch.":"The rally concludes with the ball going out of bounds after the opponent's touch.",
        "the opponent's block was successful, and the ball dropped in our side of the court":"The rally concludes with the opponent's successful block, and the ball drops on our side of the court.",
        "the opponent's block was successful, and the ball dropped in our side of the court.":"The rally concludes with the opponent's successful block, and the ball drops on our side of the court.",
        "the opponent's block was unsuccessful, and the ball dropped in their side of the court":"The rally concludes with the opponent's block failing, and the ball drops on their side of the court.",
        "the opponent's block was unsuccessful, and the ball dropped in their side of the court.":"The rally concludes with the opponent's block failing, and the ball drops on their side of the court.",
        "the opponent's block was unsuccessful, and the ball landed out of bounds":"The rally concludes with the opponent's block failing, and the ball landing out of bounds.",
        "the opponent's block was unsuccessful, and the ball landed out of bounds.":"The rally concludes with the opponent's block failing, and the ball landing out of bounds.",
        "spike fault ":"The rally concludes with a spike fault.",
        "spike fault":"The rally concludes with a spike fault.",
        "service fault":"The rally concludes with a service fault.",
        "service ace":"The rally concludes with a service ace.",
        "tip":"The rally concludes with a tip.",
    }
    
    return replacement_mapping.get(description, description) 


for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    print(file_path+'\n')
    
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        
    df = pd.read_csv(file_path, encoding=result['encoding'])
    
    df.iloc[:, 2] = df.iloc[:, 2].apply(transform_description)
    
    transformed_file_path = os.path.join(folder_path, f'new_{csv_file}')
    df.to_csv(transformed_file_path, index=False,encoding='utf-8')
    
    print(f'Transformed {csv_file} and saved as {transformed_file_path}')
    
