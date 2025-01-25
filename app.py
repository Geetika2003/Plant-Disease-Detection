# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache
@st.cache(allow_output_mutation=True)
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    return model

# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            body {
                background-image: url('https://wallup.net/wp-content/uploads/2019/09/10430-green-nature-leaves-plants-sunlight.jpg');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                color: white;
            }
            /* Animation for title */
            .animated-title {
                animation: fadeIn 3s ease-in-out;
            }
            /* Fade-in animation */
            @keyframes fadeIn {
                0% {
                    opacity: 0;
                    transform: translateY(-30px);
                }
                100% {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            @keyframes slideIn {
                0% {
                    opacity: 0;
                    transform: translateX(-100%);
                }
                100% {
                    opacity: 1;
                    transform: translateX(0);
                }
             }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Loading the Model
model = load_model('model.h5')

# Title and Description with animation
st.markdown('<h1 class="animated-title">Plant Disease Detection</h1>', unsafe_allow_html=True)
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not")

# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose an Image file", type=["png", "jpg"])

# If there is an uploaded file, start making predictions
if uploaded_file is not None:
    # Display progress and text
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    i = 0
    
    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    
    # Resize and display the image
    st.image(np.array(Image.fromarray(np.array(image)).resize((700, 400), Image.Resampling.LANCZOS)), width=None)
    my_bar.progress(i + 40)
    
    # Cleaning the image
    image = clean_image(image)
    
    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(i + 30)
    
    # Making the results
    result = make_results(predictions, predictions_arr)
    
    # Removing progress bar and text after prediction done
    my_bar.progress(i + 30)
    progress.empty()
    i = 0
    my_bar.empty()
    
    # Show the results with animation
    st.markdown(f'<p class="animated-result">The plant {result["status"]} with {result["prediction"]} prediction.</p>', unsafe_allow_html=True)
