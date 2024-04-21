import os
import pandas as pd

df = pd.read_csv(r'D:/dataset/Final_csv.csv')

output_folder = r'D:/dataset/manifest-1694365597056/Final_Dataset_cancer/Label'

os.makedirs(output_folder, exist_ok=True)

def to_yolo(row):
    x = row['X']  
    y = row['Y']  
    width = row['Width']
    height = row['Height']
    label = 0 if row['Class'] == 'benign' else 1  
    return f"{label} {x} {y} {width} {height}"

df['yolo_format'] = df.apply(to_yolo, axis=1)

for index, row in df.iterrows():
    yolo_data = row['yolo_format']
    filename = os.path.splitext(row['Filename'])[0] + '.txt'  
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'w') as file:
        file.write(yolo_data)
