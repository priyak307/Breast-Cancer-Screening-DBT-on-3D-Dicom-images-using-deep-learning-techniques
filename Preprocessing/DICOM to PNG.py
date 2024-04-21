import os
import pydicom
import re
import numpy as np
from PIL import Image

input_path = r'D:\dataset\manifest-1694365597056\Breast-Cancer-Screening-DBT'
output_path = r'D:\dataset\manifest-1694365597056\Output'

def dicom_to_png(pixel_array, png_path):
    image = pixel_array.astype(np.int16)
    image = (np.maximum(image, 0) / image.max()) * 65535.0
    image = np.uint16(image)
    img = Image.fromarray(image, mode='I;16')
    img.save(png_path)

def convert_recursive(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)
                dcm = pydicom.dcmread(dcm_path)
                parent_folder = dcm.PatientID + " " + dcm.ViewPosition
                if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames > 1:
                    for i in range(dcm.NumberOfFrames):
                        png_name = "{}frame{}.png".format(parent_folder, i+1)
                        png_path = os.path.join(output_path, png_name)
                        dicom_to_png(dcm.pixel_array[i], png_path)
                else:
                    png_name = "{}.png".format(parent_folder)
                    png_path = os.path.join(output_path, png_name)
                    dicom_to_png(dcm.pixel_array, png_path)

convert_recursive(input_path, output_path)
