import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths
input_dir = r"C:\Users\Rahul Patel\OneDrive\Documents\pythonimageprocessing\dataset_folder"
output_dir = r"C:\Users\Rahul Patel\OneDrive\Documents\pythonimageprocessing\data"

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Ensure output directories exist
for folder in ["train", "validation", "test"]:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Get all class folders (A+, A-, etc.)
class_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for class_folder in class_folders:
    class_path = os.path.join(input_dir, class_folder)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Shuffle images before splitting
    random.shuffle(images)

    # Split into train, validation, and test sets
    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Copy images to respective folders
    for image_list, folder in zip([train_images, val_images, test_images], ["train", "validation", "test"]):
        class_output_path = os.path.join(output_dir, folder, class_folder)
        os.makedirs(class_output_path, exist_ok=True)
        for img in image_list:
            shutil.copy(os.path.join(class_path, img), os.path.join(class_output_path, img))

    print(f"✅ Split {class_folder}: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")

print("🎯 Dataset split completed successfully!")
