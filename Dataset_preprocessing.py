import json
import os
import pickle
import numpy as np



def preprocessing(folder_path, max_lenght):
    data = []
    subfolder_names = os.listdir(folder_path)
    for subfolder_name in subfolder_names:
        subfolder_name = subfolder_name.replace("\\", "").replace("/", "")
        json_files = os.path.join(folder_path, subfolder_name)
        for files in os.listdir(json_files):
            if files.endswith(".json"):
                file = os.path.join(json_files, files)
                # print(file)
                try:
                    with open(file, 'r') as json_file:
                        
                        json_data = json.load(json_file)
                        
                        if json_data.get("label") is None:
                            # print(f'{file} has no label')
                            continue
                        
                        label = int(json_data["label"])
                       
                        frame_pos3d = json_data.get("frame_pos3d", {})
                        
                        if len(frame_pos3d) >= max_lenght:
                            all_positions = [frame_pos3d[str(frame_id)] for frame_id in frame_pos3d]
                            all_positions = frame_pos3d[-max_lenght:]
                            print('frame_length > max_length')
                        else:
                            new_frame_pos3d = introplate_data(frame_pos3d, max_lenght)
                            all_positions = [new_frame_pos3d[str(frame_id)] for frame_id in new_frame_pos3d]
                            print('introplate done')
                            
                        print(len(all_positions))
                        data.append((label, all_positions))
                except:
                    print(f'{file} json file open failed')
                    continue
    return data
    
    
     
        
def introplate_data(data, max_length):
    frame_ids = sorted(map(int, data.keys())) 
    
    if not frame_ids:
        frame_ids.append(0)
    
    new_data = {}
    
    for i in range(frame_ids[0], frame_ids[-1] + 1):
        new_data[str(i)] = data.get(str(i), [0, 0, 0])
        if len(new_data) == max_length:
            
            return new_data
    
    while len(new_data) < max_length:
        new_frame_id = int(list(new_data.keys())[-1]) + 1
        new_data[str(new_frame_id)] = [0, 0, 0]
        
    return new_data
     
def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
     
def main():
    types = ["train", "test", "valid"]
    max_length = [2400,500,400,300]
    aug_or_not = ["", "aug"]
    
    for aug in range(len(aug_or_not)):
        for types_num in range(len(types)):
            if aug == 0:
                folder_path = f'./data/{types[types_num]}/trajectory'
            else:
                folder_path = f'./data/{types[types_num]}/trajectory/{aug_or_not[aug]}'
            
            for i in range(len(max_length)):
                if aug == 0:
                    output_file = f'./dataset/{types[types_num]}/{types[types_num]}_last{max_length[i]}.pkl'
                else:
                    output_file = f'./dataset/{types[types_num]}/{types[types_num]}_last{max_length[i]}_{aug_or_not[aug]}.pkl'
                print(f"Start preprocessing {types[types_num]} length {max_length[i]} data \n")
                result = preprocessing(folder_path, max_length[i])
                print(f"Preprocessing {types[types_num]} length {max_length[i]} data done \n")
                print(len(result))
                save_data(result, output_file)
                print("Data saved")
    
    
    fuck = False
    
    
    for aug in range(len(aug_or_not)):
        for types_num in range(len(types)):
            for i in range(len(max_length)):
                if aug == 0:
                    output_file = f'./dataset/{types[types_num]}/{types[types_num]}_last{max_length[i]}.pkl'
                else:
                    output_file = f'./dataset/{types[types_num]}/{types[types_num]}_last{max_length[i]}_{aug_or_not[aug]}.pkl'
                
                print(f"Start checking {types[types_num]} {aug_or_not[aug]}  length {max_length[i]} data \n")
                with open(output_file, 'rb') as f:
                
                    data = pickle.load(f)
                    for j in range(len(data)):
                        if len(data[j][1]) != max_length[i]:
                            print(j)
                            print(len(data[j][1]))
                            fuck = True

                    if not fuck:
                        print("Data check is fine")
    
if __name__ == "__main__":
    main()