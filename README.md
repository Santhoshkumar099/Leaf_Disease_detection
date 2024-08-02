# Leaf Disease Detection

This project implements a deep learning model to detect diseases in plant leaves using image classification techniques.

## Project Overview

The Leaf Disease Detection project aims to assist in the early identification of plant diseases by analyzing images of plant leaves. It uses a Convolutional Neural Network (CNN) to classify leaf images into different disease categories.

## Features

- Image classification of plant leaves into 38 disease categories
- Custom CNN architecture optimized for performance on limited GPU resources
- Data augmentation to improve model generalization
- Detailed model evaluation including accuracy, loss, confusion matrix, and classification report
- Streamlit web application for easy interaction with the model

## Installation

1. Clone this repository:
   git clone https://github.com/yourusername/leaf-disease-detection.git
2. Install the required packages:
    pip install -r requirements.txt
## Usage

 ### Training the Model

 Prepare your dataset in the following structure:
   dataset/
      ├── train/
      │   ├── class1/
      │   ├── class2/
      │   └── ...
      └── test/
      ├── class1/
      ├── class2/
      └── ...
  ### Running the Streamlit App

1. Ensure you have the trained model (`leaf_disease_model.h5`) in the correct location.
2. Run the Streamlit app: streamlit run app.py
3. Open the provided URL in your web browser.
4. Upload an image of a leaf to get the disease prediction.

## Model Performance

- Training Accuracy: 90.9
- Test Accuracy: 89.4

For detailed performance metrics, please refer to the model evaluation section in the code.

## Future Improvements

- Experiment with transfer learning using pre-trained models
- Implement more advanced data augmentation techniques
- Collect and incorporate more diverse leaf images to improve model generalization

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Kaggle Leaf Disease Detection Dataset](https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset)
- Thanks to the TensorFlow and Streamlit communities for their excellent tools and documentation.    
