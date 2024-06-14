import yaml

# Define the paths to the training and validation images and annotations
data = {
    'train': r'D:/dataset/manifest-1694365597056/Final_Dataset_cancer/Images/train',  # Path to training images
    'val': r'D:/dataset/manifest-1694365597056/Final_Dataset_cancer/Images/val',  # Path to validation images
    'test': r'D:/dataset/manifest-1694365597056/Final_Dataset_cancer/Images/test',  # Optional: Path to test images
    'nc': 2,  # Number of classes
    'names': ['benign', 'cancer']  # Class names
}

# Define the path to the .yaml file
yaml_file = r'D:/dataset/manifest-1694365597056/Final_Dataset_cancer/data.yaml'

# Save the data dictionary to a .yaml file
with open(yaml_file, 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"YAML file saved to {yaml_file}")
