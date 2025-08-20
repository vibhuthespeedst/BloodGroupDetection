import pickle
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Corrected dataset path
train_dir = "C:/Users/Rahul Patel/OneDrive/Documents/pythonimageprocessing/data/train"

# Check if the directory exists
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"❌ The path '{train_dir}' does not exist. Please check your dataset location.")

# ImageDataGenerator to infer class labels
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Save class indices to a file
with open("class_indices.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)

print("✅ class_indices.pkl has been successfully created!")
