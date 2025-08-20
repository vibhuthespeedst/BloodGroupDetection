import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, test_data_dir):
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

if __name__ == '__main__':
    evaluate_model('blood_group_model.h5', 'data/test/')
