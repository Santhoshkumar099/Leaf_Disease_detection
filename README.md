# Leaf Disease Detection🍃

This project is a web application that uses machine learning to detect diseases in plant leaves from uploaded images. It's built with Python, TensorFlow, and Streamlit, and has been deployed on Hugging Face Spaces.
### App Live : https://huggingface.co/spaces/SanthoshKumar99/Leaf_disease_detection

## Features

- Upload an image of a plant leaf
- Predict the disease (if any) affecting the leaf

## Technologies Used

- Python
- TensorFlow
- Streamlit
- NumPy
- Streamlit Option Menu
- keras

## How It Works

1. The application uses a pre-trained convolutional neural network (CNN) model to classify leaf images.
2. Users can upload an image through the web interface.
3. The model processes the image and predicts the disease (or health) of the leaf.
4. The result is displayed to the user.

## Model Details

- The model can identify 38 different classes of leaf conditions, including various diseases and healthy states.
- It was trained on a dataset of leaf images, achieving high accuracy in disease detection.

## Usage

To run this project locally:

1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Deployment

This project is deployed on Hugging Face Spaces. You can access the live application [here](https://huggingface.co/spaces/SanthoshKumar99/Leaf_disease_detection).

## Reference
* [Python Documentation](https://docs.python.org/3/)
* [TenssorFlow Documentation](https://www.tensorflow.org/api_docs)
* [Scikit-learn Documentation](https://scikit-learn.org/0.21/index.html)
* [Sequential Documentation](https://keras.io/guides/sequential_model/)
* [Streamlit Documentation](https://docs.streamlit.io/)

## About the Developer

This project was developed by Santhosh Kumar M . You can find more about the developer here:

- GitHub: [https://github.com/Santhoshkumar099](https://github.com/Santhoshkumar099)
- LinkedIn: [https://www.linkedin.com/in/santhosh-kumar-2040ab188/](https://www.linkedin.com/in/santhosh-kumar-2040ab188/)
- Email: sksanthoshhkumar99@gmail.com

## Future Improvements

- Expand the dataset to include more plant species and diseases
- Implement real-time disease detection using a device camera
- Add detailed information about each detected disease and treatment recommendations

## Contributing

Contributions, issues, and feature requests are welcome.

## License

[MIT](https://choosealicense.com/licenses/mit/)
