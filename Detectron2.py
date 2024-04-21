import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer

# Set up logger
setup_logger()

# Load your dataset (images and annotations)
dataset_directory = '/content/BB_Resized'
annotations_file = '/content/drive/MyDrive/annotations_resized.json'

# Load the JSON file containing annotations
with open(annotations_file, 'r') as json_file:
    annotations_data = json.load(json_file)

# Register your dataset
def get_dataset_dicts(annotations_data, dataset_directory):
    dataset_dicts = []
    for image_filename, annotation_info in annotations_data.items():
        record = {}
        image_file_path = os.path.join(dataset_directory, image_filename)
        height, width = cv2.imread(image_file_path).shape[:2]
        record["file_name"] = image_file_path
        record["image_id"] = image_filename  # Use image filename as the image_id
        record["height"] = height
        record["width"] = width
        objs = []
        bbox = annotation_info["bbox"]
        class_label = annotation_info["Class"]
        category_id = 0  # Default category_id for "benign"
        if class_label == "cancer":
            category_id = 1  # Change category_id for "cancer"
        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYWH_ABS,  # Assuming your bbox format is [x, y, width, height]
            "category_id": category_id
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Clear the existing dataset registry
DatasetCatalog.clear()
dataset_dicts = get_dataset_dicts(annotations_data, dataset_directory)  # Define dataset_dicts here
DatasetCatalog.register("my_breast_dataset_one", lambda: dataset_dicts)

# Create a new metadata catalog entry for the modified dataset
MetadataCatalog.get("my_breast_dataset_one").set(thing_classes=["benign", "cancer"])

# Define and configure the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_breast_dataset_one",)
cfg.DATASETS.TEST = ()  # No test dataset for now
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Change this to 2 for "benign" and "cancer"

# Set up training output directory
output_dir = "/content/drive/MyDrive/output_new"
os.makedirs(output_dir, exist_ok=True)
cfg.OUTPUT_DIR = output_dir

# Create a trainer instance
trainer = DefaultTrainer(cfg)

# Train the model
trainer.resume_or_load(resume=False)
trainer.train()

# Save the final trained model (weights) in HDF5 format
output_model_path = os.path.join(output_dir, "final_model.h5")
# Save the model state_dict to an HDF5 file
torch.save(trainer.model.state_dict(), output_model_path)
print(f"Final trained model saved to {output_model_path}")

# Perform evaluation on the validation dataset (if available)
evaluator = COCOEvaluator("my_breast_dataset_one", cfg, False, output_dir="/content/drive/MyDrive")
val_loader = build_detection_test_loader(cfg, "my_breast_dataset_one")
inference_on_dataset(trainer.model, val_loader, evaluator)