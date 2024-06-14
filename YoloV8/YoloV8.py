import os
from ultralytics import YOLO

# Path to the YAML file
data_path = r'D:/dataset/manifest-1694365597056/Final_Dataset_cancer/data.yaml'

# Path to save the model
model_output_path = r'D:/dataset/manifest-1694365597056/Final_Dataset_cancer/yolov8_model'

# Create the output directory if it doesn't exist
os.makedirs(model_output_path, exist_ok=True)

# Initialize the model
model = YOLO('yolov8n.pt')  # You can use different pre-trained models, e.g., yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data=data_path,
    epochs=50,  # Number of training epochs
    batch=16,  # Batch size
    imgsz=512,  # Image size
    project=model_output_path,
    name='yolov8_model',
    cache=True  # Cache images for faster training
)

# Save the final model
model.save(os.path.join(model_output_path, 'final_yolov8_model.pt'))

print(f"Model saved to {model_output_path}")
