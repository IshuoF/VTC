import csv
import os
import json
import matplotlib.pyplot as plt

# Load the dataset
folder_path = './dataset/trajectory_text_dataset/revise'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Load json 
with open('classification_types.json') as f:
    categories = json.load(f)

contribution_dict = {category: 0 for category in categories}
uncategorized_count = 0
no_use_count = 0
tip_count = 0
tip_records = []
total_count = 0

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    print(file_path+'\n')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        
        for row in reader:
            description = row[2]
            column_0 = row[0]
            total_count += 1
            
            matched_category = None
            for category, keywords in categories.items():
                for keyword in (str(keywords) if isinstance(keywords, int) else keywords):  # 将关键词转换为字符串
                    if category in description:
                        matched_category = category
                        contribution_dict[category] += 1
                        break
                    
            if "no use" in description:
                no_use_count += 1
                matched_category = "no use"
            
            if "tip" in description:
                tip_count += 1
                matched_category = "tip"
                tip_records.append((column_0, description))
                    
            if matched_category is None:
                uncategorized_count += 1
                print(f'Uncategorized: {column_0}')
    
for category, count in contribution_dict.items():
    print(f'{category}: {count} contributions')

print(f'No Use: {no_use_count} descriptions')
print(f'Tip: {tip_count} descriptions')
for record in tip_records:
    print(f'File: {record[0]}, Description: {record[1]}')
    
print(f'Total count: {total_count}')


categories_pie = list(contribution_dict.keys())
counts_pie = list(contribution_dict.values())

plt.pie(counts_pie, labels=categories_pie,  startangle=90)

plt.text(1.2, -1.2, f'Total Count: {total_count}', horizontalalignment='center', verticalalignment='center', fontweight='bold')
plt.text(1.2, -1.8, f'No Use: {no_use_count}', horizontalalignment='center', verticalalignment='center', fontweight='bold')
plt.text(1.2, -2.4, f'Tip: {tip_count}', horizontalalignment='center', verticalalignment='center', fontweight='bold')
plt.title('Contribution Distribution by Categories')
plt.show()