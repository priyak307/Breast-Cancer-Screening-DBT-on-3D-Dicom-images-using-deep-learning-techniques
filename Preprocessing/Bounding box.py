import os
import cv2
import csv

image_folder = r'D:/dataset/manifest-1694365597056/Output'
csv_file = r'D:/dataset/data_orignal.csv'
output_file = r'D:/dataset/output.csv'
matched_images = []

with open(csv_file) as f:
    reader = csv.reader(f)
    csv_data = list(reader)[1:]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'X', 'Y', 'Width', 'Height', 'Class'])
        
        for row in csv_data:
            csv_filename = row[0]
            csv_filename = os.path.splitext(csv_filename)[0]
            match_found = False
            
            for img_file in os.listdir(image_folder):
                if img_file.startswith(csv_filename):
                    matched_images.append(img_file)
                    print("Matched image:", img_file)
                    match_found = True
                    img_path = os.path.join(image_folder, img_file)
                    img = cv2.imread(img_path)
                    x = int(row[1])
                    y = int(row[2])
                    width = int(row[3])
                    height = int(row[4])
                    scale_factor = 512 / max(img.shape[:2])
                    new_width = int(img.shape[1] * scale_factor)
                    new_height = int(img.shape[0] * scale_factor)
                    resized_img = cv2.resize(img, (new_width, new_height))
                    new_x = int(x * scale_factor)
                    new_y = int(y * scale_factor)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    writer.writerow([img_file, new_x, new_y, new_width, new_height, row[5]])

            if not match_found:
                print(f"No match found for {csv_filename}")

print("Matched images:", matched_images)
