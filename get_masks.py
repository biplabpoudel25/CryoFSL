import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm


dataset = '10028'
base_dir = './CryoPPP_dataset'
mrc_folder = os.path.join(base_dir, dataset, 'micrographs')
true_dir = os.path.join(base_dir, dataset, 'ground_truth', 'particle_coordinates')
output_folder = os.path.join(base_dir, dataset, 'outputs')

image_output_folder = os.path.join(output_folder, 'images')
mask_output_folder = os.path.join(output_folder, 'masks')
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(mask_output_folder, exist_ok=True)

target_size = (1024, 1024)

img_files = [f for f in os.listdir(mrc_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(img_files)} images in {mrc_folder}")

for filename in tqdm(img_files, desc="Processing images", unit="img"):
    f_name = os.path.splitext(filename)[0]
    t_path = f_name + '.csv'
    true_csv = os.path.join(true_dir, t_path)
    img_path = os.path.join(mrc_folder, filename)

    image_t = Image.open(img_path).convert("L")
    image = np.array(image_t, dtype=np.float32)

    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min) * 255.0
    else:
        image = np.zeros_like(image)
    image = image.astype(np.uint8)

    mask = np.zeros_like(image, dtype=np.uint8)

    if os.path.exists(true_csv):
        true_df = pd.read_csv(true_csv)
        for _, row in true_df.iterrows():
            x = int(row['X-Coordinate'])
            y = int(row['Y-Coordinate'])
            diameter = int(row['Diameter'])
            radius = int(diameter / 2)
            cv2.circle(mask, (x, y), radius, 255, -1)

#    mask = np.rot90(mask.T)
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    save_filename = f_name + '.png'
    image_output_path = os.path.join(image_output_folder, save_filename)
    mask_output_path = os.path.join(mask_output_folder, save_filename)

    cv2.imwrite(image_output_path, image_resized)
    cv2.imwrite(mask_output_path, mask_resized)

print("Finished saving images and masks!")