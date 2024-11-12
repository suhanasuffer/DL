import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import h5py
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model
import time 


# Function to create the autoencoder model with input shape (128, 128, 3)
def create_autoencoder_model(input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape, name="input_layer_7")
    # Encoding layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # Decoding layers
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded)

# Load models with their respective input shapes
autoencoder = create_autoencoder_model()
# Open the .h5 file to load weights layer by layer
with h5py.File('C:/Users/91883/OneDrive/Desktop/DL_Streamlit/enhanced_autoencoder_model_2.h5', 'r') as f:
    for layer in autoencoder.layers:
        if layer.name in f:
            try:
                # Load weights for each layer
                layer_weights = [f[layer.name][weight_name][()] for weight_name in f[layer.name]]
                
                # Check if weights match in shape
                if layer.get_weights() and all(w.shape == lw.shape for w, lw in zip(layer.get_weights(), layer_weights)):
                    layer.set_weights(layer_weights)
                    print(f"Loaded weights for layer {layer.name}")
                else:
                    print(f"Skipping layer {layer.name} due to shape mismatch.")
            except Exception as e:
                print(f"Could not load weights for layer {layer.name}: {e}")



#majorminor
def create_damage_classification_model(input_shape=(224, 224, 3)):
    input_img = Input(shape=input_shape, name="input_layer")
    
    # Define the architecture similar to the original model
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2d_1")(input_img)
    x = MaxPooling2D((2, 2), padding='same', name="max_pooling2d_1")(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2d_2")(x)
    x = MaxPooling2D((2, 2), padding='same', name="max_pooling2d_2")(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="conv2d_3")(x)
    x = MaxPooling2D((2, 2), padding='same', name="max_pooling2d_3")(x)
    
    # Flatten and add Dense layers based on original model
    x = Flatten()(x)
    x = Dense(512, activation='relu', name="dense_1")(x)
    output = Dense(1, activation='sigmoid', name="output")(x)  # Assuming binary classification
    
    # Create the model
    damage_classification_model = Model(input_img, output)
    return damage_classification_model

# Recreate the model
damage_model = create_damage_classification_model()

# Path to weights file
weight_file = 'C:/Users/91883/OneDrive/Desktop/DL_Streamlit/solar_panel_damage_model.h5'

# Load weights incrementally and handle shape mismatches
with h5py.File(weight_file, 'r') as f:
    for layer in damage_model.layers:
        if layer.name in f:
            try:
                layer_weights = [f[layer.name][weight_name][()] for weight_name in f[layer.name]]
                layer.set_weights(layer_weights)
                print(f"Loaded weights for layer {layer.name}")
            except Exception as e:
                print(f"Could not load weights for layer {layer.name}: {e}")
        else:
            print(f"No weights found for layer {layer.name}")

print("Model weights loaded with shape mismatches handled where possible.")


# Adaptive threshold and squiggliness threshold values
adaptive_threshold = 0.085
squiggliness_threshold = 2000

# Preprocess functions for each model's input shape
def preprocess_for_autoencoder(image):
    img = cv2.resize(np.array(image), (128, 128)).astype('float32') / 255.0
    return np.expand_dims(img_to_array(img), axis=0)

def preprocess_for_damage_model(image):
    img = cv2.resize(np.array(image), (224, 224)).astype('float32') / 255.0
    return np.expand_dims(img_to_array(img), axis=0)

# Classification functions for each model
def classify_image(image):
    processed_image = preprocess_for_autoencoder(image)
    reconstructed_image = autoencoder.predict(processed_image)
    error = np.mean(np.abs(processed_image - reconstructed_image))
    classification = "Dusty" if error > adaptive_threshold else "Clean"
    return processed_image[0], reconstructed_image[0], classification, error

# Ensure the function signature includes all three parameters
def predict_image_class(image, model, squiggliness_threshold):
    processed_img = preprocess_for_damage_model(image)
    prediction = model.predict(processed_img)
    damage_class = "Major Damage" if prediction[0][0] > 0.5 else "Minor Damage"
    
    squiggliness_score = calculate_squiggliness_score(image)
    squiggliness_class = "Major Damage" if squiggliness_score > squiggliness_threshold else "Minor Damage"
    
    return {
        "model_prediction": damage_class,
        "squiggliness_class": squiggliness_class,
       # "squiggliness_score": squiggliness_score
    }


# Squiggliness score calculation and contour display
def calculate_squiggliness_score(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) / 1000

def display_image_with_contours(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_img = np.array(image).copy()
    cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)
    return contoured_img

# Streamlit interface
st.title("Damage Classifier")
# Example function to simulate model prediction
def heavy_processing_task():
    time.sleep(3)  # Simulate a process that takes 3 seconds
    return "Processing Complete"

# File upload section
# File upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image.convert("RGB"))  # Convert to RGB format for OpenCV compatibility
    
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run the autoencoder classification
    original, reconstructed, classification, error = classify_image(image)
    
    # Display classification result and reconstruction error
    st.write(f"**Classification Result:** {classification}")
    st.write(f"**Reconstruction Error:** {error:.4f}")

    # Run the damage model prediction
    results = predict_image_class(image, damage_model, squiggliness_threshold)
    
    # Display model prediction and squiggliness details
    st.write(f"**Damage Prediction:** {results['model_prediction']}")
    #st.write(f"**Squiggliness Classification:** {results['squiggliness_class']}")
    #st.write(f"**Squiggliness Score:** {results['squiggliness_score']:.4f}")
    
    # Display the contoured image
    contoured_img = display_image_with_contours(image)
    st.image(contoured_img, caption="Image with Detected Contours", use_container_width=True)


