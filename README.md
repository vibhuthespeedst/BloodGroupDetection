# Blood Group Detection Using Infrared Hand Images

![image](https://github.com/user-attachments/assets/5f6ada77-da41-4d15-93c7-d878126f4883)

![image](https://github.com/user-attachments/assets/310f3e27-b45c-4674-bbb0-06c6a986e121)

## Overview
This project utilizes deep learning techniques to detect blood groups from infrared images of hands. By analyzing spectroscopic features, the model classifies blood types with high accuracy.

## Features
- Automated blood group classification
- Uses infrared hand images for detection
- Deep learning-based model trained on a labeled dataset
- Flask web application for easy interaction

## Dataset
The dataset consists of infrared images categorized into different blood groups:

```
/dataset_folder
    /train
        /A Positive
        /A Negative
        /AB Positive
        /AB Negative
        /B Positive
        /B Negative
        /O Positive
    /test
    /validation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vibhuthespeedst/BloodGroupDetection.git
   cd blood-group-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```

## Model Training
To train the model, execute:
```bash
python train_model.py
```
The trained model is saved as `blood_group_model_vgg16.h5`.

## Usage
1. Start the Flask app and open the web interface.
2. Upload an infrared image of a hand.
3. The model predicts and displays the blood group.

## Screenshots
### Model Training
![Screenshot 2025-03-10 054922](https://github.com/user-attachments/assets/04ae9d30-552b-4317-b74d-0d5f16a35d77)


### Web Interface
![image](https://github.com/user-attachments/assets/a5e9f247-8fe3-4de7-bdb9-90fb1fb44260)

![image](https://github.com/user-attachments/assets/06faadac-129f-4816-ad9f-6a8e02c7d074)

![image](https://github.com/user-attachments/assets/db1d3ec2-e1e9-4536-9fbc-c42035d42319)




## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- Flask

## Results
The model achieved an accuracy of **93%** on the test dataset. Below is the confusion matrix:

![Screenshot 2025-03-10 054656](https://github.com/user-attachments/assets/13dc3b46-24c4-4fa9-961d-5cc29d0a2acf)


## Contributors
-**Vibhu Mishra** - Developer
- **Rahul Patel** - Developer

Feel free to contribute to this project by submitting issues or pull requests!
