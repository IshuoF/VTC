import csv 
import json
import os
import chardet

def label_to_json(csv_path, json_path):
    
    csv_path = csv_path + '.csv'
    csv_path = os.path.normpath(csv_path)
    
    # print(csv_path+'\n')
    
    with open(csv_path, 'rb') as file:
        result = chardet.detect(file.read())
        # print(csv_path+" "+ str(result)+ '\n')
    
    with open(csv_path, 'r',encoding=result['encoding']) as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_data = list(csv_reader)
        # print(csv_data)
        
    with open("classification_categories.json", 'r') as categories_file:
        categories = json.load(categories_file)
    
    for row in csv_data:
        video_name = row[0].split('.')[0]
        label = row[2]
        
        label_type =''
        json_path_found =''
        for category, category_numbers in categories.items():
            for category_numbers in (str(category_numbers) if isinstance(category_numbers, int) else category_numbers):
                if label in category:
                    label_type = category_numbers
                    # print(f'{video_name}: {label} {label_type} \n')
                    
                    
                    aug_video_names = os.listdir(json_path)
                    
                    aug_video_name = next((file for file in aug_video_names if video_name in file), None)
                    # print(aug_video_name)
                    json_path_found = os.path.join(json_path, str(aug_video_name))
                    # print(json_path_found)
                    
                    if os.path.exists(json_path_found):
                        
                        try:
                            with open(json_path_found , 'r') as json_file:
                                json_data = json.load(json_file)
                                
                            json_data["label"] = label_type
                            
                            with open(json_path_found, 'w') as json_file:
                                json.dump(json_data, json_file, indent=2, ensure_ascii=False)
                                
                            print(f'{json_path_found} json file dump\n')
                            
                        except:
                            print(f'{json_path_found} json file dump failed\n')
                break      
max_frame_length
                
 
def main(csv_folder, json_folder):
    folder_names = os.listdir(json_folder)
    max = 0 
    for folder_name in folder_names:
        
        folder_name = folder_name.replace("\\", "").replace("/", "")
        csv_path = os.path.join(csv_folder, folder_name)
        csv_path = csv_path.replace('_aug','')
        json_path = os.path.join(json_folder, folder_name)
        label_to_json(csv_path, json_path)
        
    

if __name__ == '__main__':
    
    csv_folder = './data/trajectory_text_data/revise'
    
    json_folder = './data/valid/trajectory/aug'
    
    main(csv_folder, json_folder)
    